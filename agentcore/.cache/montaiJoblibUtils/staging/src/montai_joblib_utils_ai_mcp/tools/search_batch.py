"""Search AWS Batch job runs (executions) and, as a fallback, job definitions.

Two public entry-points
-----------------------
search_batch_compute  — discover whether a compute resource *exists*.
    Pass 1: RUNNING fan-out (46 calls, ~1 s) — is this job currently executing?
    Pass 2: describe_job_definitions exact/page scan — does the definition exist?

search_batch_runs     — retrieve recent run history (SUCCEEDED/FAILED/RUNNING).
    Fans out list_jobs across all enabled queues for the requested statuses.
    Queue list is cached via _resource_cache so describe_job_queues is called
    only once per process lifetime (background thread keeps it fresh).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
from botocore.config import Config
from strands import tool

from montai_joblib_utils_ai_mcp.tools import _resource_cache as cache
from montai_joblib_utils_ai_mcp.tools._match import matches_query, tag_values

logger = logging.getLogger(__name__)

# Only RUNNING for the compute-discovery fan-out. Checking one status keeps API
# calls to 1×N (46 queues → 46 calls) and avoids Batch ListJobs rate-limit issues.
_ACTIVE_RUN_STATUSES = ["RUNNING"]

# Default statuses for run-history search. SUCCEEDED + FAILED cover completed work;
# RUNNING covers anything executing right now.
_DEFAULT_HISTORY_STATUSES = ["SUCCEEDED", "FAILED", "RUNNING"]

_CACHE_KEY_QUEUES = "batch_queues"


def _list_all_queues(client: Any) -> list[dict[str, Any]]:
    """Return all job queue objects from describe_job_queues (all pages)."""
    queues: list[dict[str, Any]] = []
    paginator = client.get_paginator("describe_job_queues")
    for page in paginator.paginate():
        queues.extend(page.get("jobQueues", []))
    return queues


def _get_queues(client: Any, region_name: str | None, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return the queue list from cache, or fetch and cache it if missing."""
    key_region = region_name or "default"
    cached = cache.get(_CACHE_KEY_QUEUES, key_region, force_refresh=force_refresh)
    if cached is None:
        cached = _list_all_queues(client)
        cache.set(
            _CACHE_KEY_QUEUES,
            key_region,
            cached,
            refresh_fn=lambda: _list_all_queues(
                boto3.client(
                    "batch",
                    region_name=region_name,
                    config=Config(max_pool_connections=50, retries={"mode": "adaptive"}),
                )
            ),
        )
    return cached


def _resource_map(requirements: list[dict] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for req in requirements or []:
        rtype = req.get("type") or req.get("Type")
        value = req.get("value") or req.get("Value")
        if rtype and value is not None:
            out[str(rtype)] = str(value)
    return out


def _run_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Summarise a Batch job run (from list_jobs / describe_jobs)."""
    container = job.get("container") or {}
    log_stream = container.get("logStreamName") or None
    return {
        "name": job.get("jobName", ""),
        "job_id": job.get("jobId"),
        "job_definition": job.get("jobDefinition"),
        "queue": job.get("jobQueue"),
        "status": job.get("status"),
        "status_reason": job.get("statusReason") or None,
        "created_at": job.get("createdAt"),
        "started_at": job.get("startedAt"),
        "stopped_at": job.get("stoppedAt"),
        "log_stream_name": log_stream,
        "log_group": "/aws/batch/job" if log_stream else None,
        "job_type": "batch",
        "result_source": "run",
    }


def _jd_summary(jd: dict[str, Any]) -> dict[str, Any]:
    """Summarise a Batch job definition (from describe_job_definitions)."""
    container = jd.get("containerProperties") or {}
    return {
        "name": jd.get("jobDefinitionName", ""),
        "arn": jd.get("jobDefinitionArn"),
        "revision": jd.get("revision"),
        "status": jd.get("status"),
        "image": container.get("image") or None,
        "command": container.get("command") or None,
        "resources": _resource_map(container.get("resourceRequirements")) or None,
        "platform_capabilities": jd.get("platformCapabilities"),
        "tags": tag_values(jd.get("tags")) or None,
        "job_type": "batch",
        "result_source": "job_definition",
    }


def _list_jobs_for_queue(
    client: Any,
    queue: str,
    status: str,
    query: str | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Call list_jobs for one (queue, status) pair and return job summaries.

    list_jobs does not include jobQueue in its response objects, so we stamp it
    onto every result here while we still have the queue name in scope.
    """
    kwargs: dict[str, Any] = {
        "jobQueue": queue,
        "jobStatus": status,
        "maxResults": min(max_results, 100),
    }
    if query and query.strip():
        kwargs["filters"] = [{"name": "JOB_NAME", "values": [f"*{query.strip()}*"]}]
    try:
        resp = client.list_jobs(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("list_jobs failed for queue=%s status=%s: %s", queue, status, exc)
        return []
    jobs = resp.get("jobSummaryList", [])
    for job in jobs:
        job.setdefault("jobQueue", queue)
    return jobs


def _search_runs(
    client: Any,
    query: str | None,
    max_results: int,
    queues: list[str],
) -> list[dict[str, Any]]:
    """Pass 1: search currently-RUNNING jobs via list_jobs fan-out across all enabled queues.

    Only checks RUNNING status (not SUBMITTED/PENDING/STARTING/SUCCEEDED/FAILED).
    Transient pre-run states last seconds and are rarely useful for discovery.
    Completed states would require many more API calls and trigger rate-limits.
    46 queues × 1 status = 46 calls, well within Batch ListJobs rate limits.
    """
    if not queues:
        return []

    work = [(q, s) for q in queues for s in _ACTIVE_RUN_STATUSES]
    raw: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(len(work), 16)) as pool:
        futures = {
            pool.submit(_list_jobs_for_queue, client, q, s, query, max_results): (q, s)
            for q, s in work
        }
        for fut in as_completed(futures):
            raw.extend(fut.result())

    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in sorted(raw, key=lambda j: j.get("createdAt") or 0, reverse=True):
        name = job.get("jobName", "")
        if name in seen:
            continue
        if not matches_query(query, name, job.get("jobDefinition", ""), job.get("status", "")):
            continue
        seen.add(name)
        runs.append(_run_summary(job))
        if len(runs) >= max_results:
            break
    return runs


def _search_definitions(
    client: Any,
    query: str | None,
    status: str,
    max_results: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Pass 2 fallback: scan job definitions.

    Tries exact jobDefinitionName first (O(1)), then falls back to a bounded
    page scan.  Returns (matches, truncated).

    Deduplication strategy: seen_names tracks every definition name we encounter,
    not just matching ones. AWS Batch returns all revisions as separate entries
    (some definitions have thousands of ACTIVE revisions). By skipping every
    name after its first appearance we process only the newest revision of each
    definition and avoid burning page budget on stale revisions.
    """
    matches: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    truncated = False

    if query and query.strip():
        try:
            exact = client.describe_job_definitions(
                jobDefinitionName=query.strip(), status=status
            )
            for jd in exact.get("jobDefinitions", []):
                summary = _jd_summary(jd)
                if summary["name"] not in seen_names:
                    seen_names.add(summary["name"])
                    matches.append(summary)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Exact job definition lookup failed: %s", exc)

    if matches:
        return matches, False

    paginator = client.get_paginator("describe_job_definitions")
    unique_names_seen = 0
    pages = 0
    for page in paginator.paginate(
        status=status,
        PaginationConfig={"MaxItems": max_pages * 100, "PageSize": 100},
    ):
        pages += 1
        for jd in page.get("jobDefinitions", []):
            name = jd.get("jobDefinitionName", "")
            if name in seen_names:
                continue
            # Mark this name seen immediately — skip all subsequent revisions
            # regardless of whether this revision matches the query. AWS Batch
            # returns newest-first, so the first entry is the current revision.
            seen_names.add(name)
            unique_names_seen += 1

            container = jd.get("containerProperties") or {}
            image = container.get("image") or ""
            command = " ".join(container.get("command") or [])
            tags = tag_values(jd.get("tags"))
            if not matches_query(query, name, image, command, *tags):
                continue
            matches.append(_jd_summary(jd))
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results or pages >= max_pages:
            if pages >= max_pages and len(matches) < max_results:
                truncated = True
            break

    logger.debug(
        "definition scan: %d pages, %d unique names, %d matches",
        pages,
        unique_names_seen,
        len(matches),
    )
    return matches, truncated


def search_batch_compute_impl(
    query: str | None = None,
    status: str = "ACTIVE",
    max_results: int = 25,
    include_queues: bool = True,
    region_name: str | None = None,
    max_pages: int = 5,
) -> dict[str, Any]:
    """Search Batch job runs first; fall back to job definitions on a miss.

    Pass 1 — list_jobs fan-out (active runs only, parallel across all queues).
    Pass 2 — describe_job_definitions (fallback, bounded page scan).
    """
    # Pool sized for the fan-out (46 queues, capped at 16 workers).
    # Adaptive retry mode automatically backs off on Batch ListJobs rate-limit errors.
    client = boto3.client(
        "batch",
        region_name=region_name,
        config=Config(max_pool_connections=50, retries={"mode": "adaptive"}),
    )
    notes: list[str] = []

    try:
        # Queue list comes from cache — describe_job_queues is only called once
        # per process lifetime and refreshed in the background.
        all_queue_objs = _get_queues(client, region_name)
        enabled_queue_names = [
            q["jobQueueName"] for q in all_queue_objs if q.get("state") == "ENABLED"
        ]

        # Pass 1: active runs via list_jobs fan-out across queues
        results = _search_runs(client, query, max_results, enabled_queue_names)
        result_source = "runs"

        # Pass 2: fall back to definitions only when runs return nothing
        truncated = False
        if not results:
            results, truncated = _search_definitions(client, query, status, max_results, max_pages)
            result_source = "job_definitions"
            if truncated:
                notes.append(
                    f"Definition scan limited to {max_pages} pages of ACTIVE job definitions; "
                    "pass job_type='batch' explicitly for a targeted search."
                )

        queues: list[dict[str, Any]] = []
        if include_queues:
            for q in all_queue_objs:
                qname = q.get("jobQueueName", "")
                if matches_query(query, qname, q.get("state"), *(tag_values(q.get("tags")))):
                    queues.append({
                        "name": qname,
                        "arn": q.get("jobQueueArn"),
                        "state": q.get("state"),
                        "priority": q.get("priority"),
                        "status": q.get("status"),
                    })

    except Exception as exc:  # noqa: BLE001
        logger.error("Batch search failed: %s", exc)
        return {
            "job_type": "batch",
            "query": query,
            "count": 0,
            "results": [],
            "queues": [],
            "error": str(exc),
        }

    out: dict[str, Any] = {
        "job_type": "batch",
        "query": query,
        "result_source": result_source,
        "count": len(results),
        "results": results[:max_results],
        "queues": queues if include_queues else [],
    }
    if notes:
        out["notes"] = notes
    return out


@tool
def search_batch_compute(
    query: str | None = None,
    status: str = "ACTIVE",
    max_results: int = 25,
    include_queues: bool = True,
) -> dict[str, Any]:
    """Search AWS Batch for heavy / GPU compute — active runs first, definitions as fallback.

    Pass 1 fans out list_jobs(jobQueue, jobStatus) in parallel across all enabled
    queues for active statuses (RUNNING/SUBMITTED/PENDING/STARTING). This reveals
    what is currently executing without requiring any queue knowledge upfront.
    Pass 2 falls back to scanning ACTIVE job definitions only when Pass 1 returns
    no hits.

    Use for names like acn_predictions, murcko_scaffolds, montai-ondemand-* —
    find what has actually been running rather than what is defined.

    Args:
        query: Case-insensitive substring matched against job/definition name,
               image, command, tags. Supports wildcards in the run search.
        status: Job definition status filter for the fallback pass (default ACTIVE).
        max_results: Cap on returned results (default 25).
        include_queues: Also return matching job queues.

    Returns:
        dict with job_type="batch", result_source ("runs" or "job_definitions"),
        count, results, and queues.
    """
    return search_batch_compute_impl(
        query=query,
        status=status,
        max_results=max_results,
        include_queues=include_queues,
    )


# ---------------------------------------------------------------------------
# Run-history search
# ---------------------------------------------------------------------------

def _enrich_with_describe(client: Any, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich run summaries with log_stream_name and job_definition via describe_jobs.

    list_jobs only returns exitCode inside container — logStreamName and
    jobDefinition ARN require a describe_jobs call.  We batch up to 100 IDs
    per call (the API limit) and merge the enriched fields back in-place.
    """
    if not runs:
        return runs
    ids = [r["job_id"] for r in runs if r.get("job_id")]
    if not ids:
        return runs

    details: dict[str, dict[str, Any]] = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        try:
            resp = client.describe_jobs(jobs=chunk)
            for j in resp.get("jobs", []):
                details[j["jobId"]] = j
        except Exception as exc:  # noqa: BLE001
            logger.debug("describe_jobs enrichment failed: %s", exc)

    for run in runs:
        detail = details.get(run.get("job_id", ""))
        if not detail:
            continue
        container = detail.get("container") or {}
        log_stream = container.get("logStreamName") or None
        run["log_stream_name"] = log_stream
        run["log_group"] = "/aws/batch/job" if log_stream else None
        run["job_definition"] = detail.get("jobDefinition") or run.get("job_definition")

    return runs


def _fanout_runs(
    client: Any,
    query: str | None,
    statuses: list[str],
    queues: list[str],
    max_per_queue: int,
) -> list[dict[str, Any]]:
    """Fan out list_jobs across all enabled queues × requested statuses in parallel.

    Returns raw job summary dicts, unsorted and not yet deduplicated.
    max_per_queue caps how many results we pull from each (queue, status) pair
    so the server-side response stays small.
    """
    if not queues:
        return []
    work = [(q, s) for q in queues for s in statuses]
    raw: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(work), 32)) as pool:
        futures = {
            pool.submit(_list_jobs_for_queue, client, q, s, query, max_per_queue): (q, s)
            for q, s in work
        }
        for fut in as_completed(futures):
            raw.extend(fut.result())
    return raw


def search_batch_runs_impl(
    query: str | None = None,
    statuses: list[str] | None = None,
    max_results: int = 20,
    max_per_queue: int = 10,
    region_name: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Search recent Batch job runs (SUCCEEDED/FAILED/RUNNING) across all queues.

    Uses a cached queue list and parallel list_jobs fan-out.  Results are sorted
    newest-first and deduplicated by job_id.

    Args:
        query: Substring / wildcard matched against job name and definition ARN.
        statuses: Which run statuses to include (default: SUCCEEDED, FAILED, RUNNING).
        max_results: Total results to return (default 20).
        max_per_queue: Max results fetched from each (queue, status) pair (default 10).
            Keeps each API response small; raise for historical depth.
        region_name: AWS region (default: env/profile default).
        force_refresh: Evict the cached queue list and re-fetch from AWS.
    """
    client = boto3.client(
        "batch",
        region_name=region_name,
        config=Config(max_pool_connections=50, retries={"mode": "adaptive"}),
    )
    statuses = statuses or _DEFAULT_HISTORY_STATUSES

    try:
        all_queue_objs = _get_queues(client, region_name, force_refresh=force_refresh)
        enabled_queues = [
            q["jobQueueName"] for q in all_queue_objs if q.get("state") == "ENABLED"
        ]

        raw = _fanout_runs(client, query, statuses, enabled_queues, max_per_queue)

        seen_ids: set[str] = set()
        results: list[dict[str, Any]] = []
        for job in sorted(raw, key=lambda j: j.get("createdAt") or 0, reverse=True):
            job_id = job.get("jobId", "")
            if job_id in seen_ids:
                continue
            name = job.get("jobName", "")
            if not matches_query(query, name, job.get("jobDefinition", ""), job.get("status", "")):
                continue
            seen_ids.add(job_id)
            results.append(_run_summary(job))
            if len(results) >= max_results:
                break

        # Enrich with log stream name and job definition ARN from describe_jobs.
        # list_jobs only returns exitCode; the rest live in the describe response.
        _enrich_with_describe(client, results)

    except Exception as exc:  # noqa: BLE001
        logger.error("Batch run search failed: %s", exc)
        return {
            "job_type": "batch",
            "query": query,
            "statuses": statuses,
            "count": 0,
            "results": [],
            "error": str(exc),
        }

    return {
        "job_type": "batch",
        "query": query,
        "statuses": statuses,
        "queues_searched": len(enabled_queues),
        "count": len(results),
        "results": results,
    }


@tool
def search_batch_runs(
    query: str | None = None,
    statuses: list[str] | None = None,
    max_results: int = 20,
    max_per_queue: int = 10,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Search recent AWS Batch job runs (executions) across all enabled queues.

    Use this when you want to see whether a job has *run* recently — not just
    whether the definition exists.  Fans out list_jobs in parallel across every
    enabled queue for the requested statuses, then returns results newest-first.

    The Batch queue list is cached in-process and refreshed automatically every
    ~4 minutes.  Pass force_refresh=True to re-fetch it immediately from AWS.

    Args:
        query: Case-insensitive substring matched against job name and definition
               ARN.  Wildcards (*) are applied server-side on the job name.
               Pass None to return recent runs across all jobs.
        statuses: Which run statuses to include.
               Default: ["SUCCEEDED", "FAILED", "RUNNING"].
               Options: SUBMITTED, PENDING, STARTING, RUNNING, SUCCEEDED, FAILED.
        max_results: Total results to return across all queues (default 20).
        max_per_queue: Max results fetched per (queue, status) pair (default 10).
               Increase to dig further back in history.
        force_refresh: Re-fetch the queue list from AWS, bypassing the cache.

    Returns:
        dict with job_type="batch", statuses, queues_searched, count, and results
        sorted newest-first.  Each result includes job_id, queue, status,
        created_at, started_at, stopped_at, and job_definition ARN.
    """
    return search_batch_runs_impl(
        query=query,
        statuses=statuses,
        max_results=max_results,
        max_per_queue=max_per_queue,
        force_refresh=force_refresh,
    )
