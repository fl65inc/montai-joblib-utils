"""Fast discovery: classify job_type, then targeted or parallel fan-out search."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from strands import tool

from montai_joblib_utils_ai_mcp.tools.classify import classify_job_type
from montai_joblib_utils_ai_mcp.tools.query_terms import extract_search_query
from montai_joblib_utils_ai_mcp.tools.search_batch import search_batch_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_compute import search_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_lambda import search_lambda_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_sagemaker import (
    _enrich_pipeline,
    search_sagemaker_compute_impl,
)
from montai_joblib_utils_ai_mcp.tools.search_stepfunctions import (
    search_stepfunctions_compute_impl,
)
from montai_joblib_utils_ai_mcp.tools.state_machine_tree import (
    describe_state_machine_tree_impl,
)
from montai_joblib_utils_ai_mcp.types import JobType

logger = logging.getLogger(__name__)

# Priority when picking a primary hit from a mixed fan-out.
_PRIMARY_RANK = {
    "sagemaker_pipeline": 0,
    "stepfunctions": 1,
    "state_machine": 1,
    "batch": 2,
    "lambda": 3,
    "sagemaker_training": 4,
    "sagemaker_hyperparameter": 5,
}


def _run_worker(key: str, fn, query: str | None) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.error("Fan-out worker %s failed: %s", key, exc)
        return {
            "job_type": key,
            "query": query,
            "count": 0,
            "results": [],
            "error": str(exc),
        }


def _fanout_workers(
    query: str | None,
    max_results: int,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Staged fan-out: fast families first; Batch only if nothing matched.

    AWS Batch ``DescribeJobDefinitions`` is extremely slow (minutes for a full
    scan). Waiting on it in every fan-out made discovery feel broken. So:

      Phase 1 (parallel): lambda ‖ sagemaker_pipeline ‖ stepfunctions
      If any hits → return immediately (skip Batch)
      Phase 2: Batch only as a fallback

    ``force_refresh`` is forwarded to cached searchers (SFN, Lambda) so they
    re-fetch from AWS instead of serving stale list data.
    """

    def _lambda() -> dict[str, Any]:
        return search_lambda_compute_impl(
            query=query, max_results=max_results, force_refresh=force_refresh
        )

    def _sagemaker() -> dict[str, Any]:
        return search_sagemaker_compute_impl(
            query=query,
            kind="pipeline",
            max_results=max_results,
            enrich=False,
        )

    def _sfn() -> dict[str, Any]:
        return search_stepfunctions_compute_impl(
            query=query,
            max_results=max_results,
            enrich=False,
            force_refresh=force_refresh,
        )

    def _batch() -> dict[str, Any]:
        return search_batch_compute_impl(
            query=query,
            max_results=max_results,
            include_queues=False,
            max_pages=1,  # fallback only — keep tiny
        )

    fast = {
        "lambda": _lambda,
        "sagemaker_pipeline": _sagemaker,
        "stepfunctions": _sfn,
    }
    by_type: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=len(fast)) as pool:
        futures = {pool.submit(_run_worker, key, fn, query): key for key, fn in fast.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            by_type[key] = fut.result()

    if any((by_type[k].get("count") or 0) > 0 for k in by_type):
        by_type["batch"] = {
            "job_type": "batch",
            "query": query,
            "count": 0,
            "results": [],
            "skipped": True,
            "note": "Skipped — fast families already returned hits (Batch list is slow).",
        }
        return by_type

    # Nothing in the fast path — pay for Batch.
    by_type["batch"] = _run_worker("batch", _batch, query)
    return by_type


def _flatten_hits(by_type: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for payload in by_type.values():
        for item in payload.get("results") or []:
            hits.append(item)
    hits.sort(key=lambda h: _PRIMARY_RANK.get(h.get("job_type", ""), 99))
    return hits


def _enrich_primary(primary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lazy-enrich the winning hit so the agent gets definition / tree detail."""
    if not primary:
        return primary
    jt = primary.get("job_type")
    name = primary.get("name")
    arn = primary.get("arn")
    try:
        if jt == "sagemaker_pipeline" and name and "definition_preview" not in primary:
            import boto3

            client = boto3.client("sagemaker")
            return _enrich_pipeline(client, name, dict(primary))
        if jt in ("stepfunctions", "state_machine") and arn and "tree" not in primary:
            tree_out = describe_state_machine_tree_impl(state_machine_arn=arn)
            enriched = dict(primary)
            enriched["tree"] = tree_out.get("tree")
            enriched["leaves"] = tree_out.get("leaves") or []
            enriched["leaf_count"] = tree_out.get("leaf_count", 0)
            enriched["status"] = tree_out.get("status") or enriched.get("status")
            return enriched
    except Exception as exc:  # noqa: BLE001
        logger.warning("Primary enrich failed: %s", exc)
    return primary


def discover_compute_impl(
    query: str,
    job_type: JobType | None = None,
    max_results: int = 10,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Classify → targeted search OR parallel fan-out across job families.

    This is the fast path the agent should call once (see SOP compute-discovery).
    Pass ``force_refresh=True`` to bypass the in-process list cache and re-fetch
    live data from AWS for SFN and Lambda.
    """
    t0 = time.perf_counter()
    classification = classify_job_type(query, job_type=job_type)
    # Strip cue words so "tractability inference pipeline" matches MedChem*Tractability*.
    search_query = extract_search_query(query)

    # High/medium confidence → one targeted call.
    if classification["known"] and classification["confidence"] in ("high", "medium"):
        jt: JobType = classification["job_type"]
        targeted = search_compute_impl(
            job_type=jt,
            query=search_query,
            max_results=max_results,
        )
        hits = list(targeted.get("results") or [])
        primary = _enrich_primary(hits[0] if hits else None)
        if primary and hits:
            hits[0] = primary
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "query": query,
            "search_query": search_query,
            "classification": classification,
            "strategy": "targeted",
            "elapsed_ms": elapsed_ms,
            "by_job_type": {jt: targeted},
            "hits": hits,
            "primary": primary,
            "count": len(hits),
        }

    # Unknown → parallel fan-out.
    by_type = _fanout_workers(search_query, max_results=max_results, force_refresh=force_refresh)
    hits = _flatten_hits(by_type)
    primary = _enrich_primary(hits[0] if hits else None)
    if primary and hits:
        hits[0] = primary
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "query": query,
        "search_query": search_query,
        "classification": classification,
        "strategy": "parallel_fanout",
        "elapsed_ms": elapsed_ms,
        "by_job_type": {
            k: {
                "job_type": v.get("job_type"),
                "count": v.get("count", 0),
                "error": v.get("error"),
                "results": v.get("results") or [],
                **(
                    {"skipped": v["skipped"], "note": v.get("note")}
                    if v.get("skipped")
                    else {}
                ),
            }
            for k, v in by_type.items()
        },
        "hits": hits,
        "primary": primary,
        "count": len(hits),
    }


@tool
def discover_compute(
    query: str,
    job_type: JobType | None = None,
    max_results: int = 10,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Discover existing compute for a workload — classify then search (fast path).

    Prefer this over calling family search tools yourself.

    Flow:
      1. Ask: do we know the job_type? (explicit arg or classifier cues)
      2. YES → one targeted search
      3. NO  → staged fan-out: lambda ‖ sagemaker_pipeline ‖ stepfunctions
               first; Batch only if those miss

    The SFN and Lambda list responses are cached in-process across calls (TTL
    default 300 s, override with ``MONTAI_RESOURCE_CACHE_TTL``). Pass
    ``force_refresh=True`` to discard stale cached lists and re-fetch from AWS
    — useful after a deploy or when you suspect the cache is stale. You can
    also call ``invalidate_resource_cache`` to pre-emptively clear the cache
    before the next search.

    Args:
        query: Workload name or distinctive substring (e.g. "tractability").
        job_type: Optional known type — skips classification when set.
        max_results: Cap per family (default 10).
        force_refresh: When True, bypass the cache and re-fetch from AWS.

    Returns:
        classification, strategy (targeted|parallel_fanout), elapsed_ms,
        primary hit, and all hits.
    """
    return discover_compute_impl(
        query=query,
        job_type=job_type,
        max_results=max_results,
        force_refresh=force_refresh,
    )
