"""
FastAPI REST server for Montai compute discovery.

Exposes all search tools as typed HTTP endpoints with full OpenAPI/Swagger docs.
Pair with on-demand-compute-ai-mcp for job submission.

Usage:
  uv run montai-joblib-utils-api
  uv run montai-joblib-utils-api --port 8082 --reload
  # then open: http://localhost:8082/docs
"""

from __future__ import annotations

import argparse
import logging
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from montai_joblib_utils_ai_mcp.tools._resource_cache import invalidate_resource_cache
from montai_joblib_utils_ai_mcp.tools.discover_compute import discover_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_batch import (
    search_batch_compute_impl,
    search_batch_runs_impl,
)
from montai_joblib_utils_ai_mcp.tools.search_lambda import search_lambda_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_sagemaker import search_sagemaker_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_stepfunctions import (
    search_stepfunctions_compute_impl,
)
from montai_joblib_utils_ai_mcp.tools.state_machine_tree import (
    describe_state_machine_tree_impl,
)
from montai_joblib_utils_ai_mcp.types import JobType

logging.basicConfig(level=logging.INFO)


class InvocationRequest(BaseModel):
    """AgentCore HTTP protocol — primary invocation payload."""

    prompt: str
    job_type: str | None = None
    max_results: int = 10
    force_refresh: bool = False


app = FastAPI(
    title="Montai Compute Discovery API",
    description=(
        "Discover existing AWS compute resources — Lambda, Batch, SageMaker, and "
        "Step Functions — without submitting jobs. All endpoints are read-only.\n\n"
        "**Quick start:** `GET /discover?query=acn_predictions`"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"], summary="Health check")
def health() -> dict[str, str]:
    """Returns `{"status": "ok"}` — use for liveness/readiness probes."""
    return {"status": "ok"}


@app.get("/ping", tags=["meta"], summary="AgentCore health probe")
def ping() -> dict[str, str]:
    """AgentCore Runtime liveness probe — must return `{"status": "Healthy"}`."""
    return {"status": "Healthy"}


# ---------------------------------------------------------------------------
# AgentCore HTTP protocol — /invocations
# ---------------------------------------------------------------------------


@app.post("/invocations", tags=["agent"], summary="AgentCore primary invocation endpoint")
async def invocations(request: Request) -> dict[str, Any]:
    """AgentCore HTTP protocol entrypoint.

    Accepts a natural-language ``prompt`` and routes it through
    ``discover_compute``.  Tolerates any Content-Type (agentcore CLI,
    boto3, curl) by parsing the raw body as JSON directly.

    Request body (JSON):
        ``{"prompt": "find batch jobs for acn_predictions"}``
        ``{"prompt": "acn", "job_type": "batch", "max_results": 5}``

    The ``agentcore invoke`` CLI wraps plain strings as
    ``{"prompt": "..."}`` automatically.  If a nested JSON object is
    detected inside ``prompt`` it is unwrapped transparently.

    Returns the same shape as ``GET /discover``.
    """
    import json as _json

    raw = await request.body()
    data: dict[str, Any] = {}
    try:
        data = _json.loads(raw)
        if not isinstance(data, dict):
            data = {"prompt": str(data)}
    except (_json.JSONDecodeError, ValueError):
        data = {"prompt": raw.decode(errors="replace")}

    # agentcore invoke '{"prompt": "acn"}' double-wraps: unwrap inner JSON
    prompt: str = str(data.get("prompt", ""))
    try:
        inner = _json.loads(prompt)
        if isinstance(inner, dict) and "prompt" in inner:
            if "job_type" not in data and "job_type" in inner:
                data["job_type"] = inner["job_type"]
            prompt = str(inner["prompt"])
    except (_json.JSONDecodeError, TypeError, ValueError):
        pass

    from montai_joblib_utils_ai_mcp.types import JOB_TYPES

    raw_jt = data.get("job_type")
    job_type_val = raw_jt if raw_jt in JOB_TYPES else None

    return discover_compute_impl(
        query=prompt,
        job_type=job_type_val,
        max_results=int(data.get("max_results", 10)),
        force_refresh=bool(data.get("force_refresh", False)),
    )


# ---------------------------------------------------------------------------
# Discover (REST entry-point)
# ---------------------------------------------------------------------------

@app.get("/discover", tags=["discover"], summary="Discover compute for a workload")
def discover(
    query: Annotated[str, Query(description="Workload name or description to search for")],
    job_type: Annotated[
        JobType | None,
        Query(description="Skip classification and go straight to this compute family"),
    ] = None,
    max_results: Annotated[int, Query(ge=1, le=100)] = 10,
    force_refresh: Annotated[
        bool,
        Query(description="Bypass the in-process cache and re-fetch lists from AWS"),
    ] = False,
) -> dict[str, Any]:
    """Classify the query, then run a targeted or parallel fan-out search.

    - **Known job_type** → single targeted AWS call (fast, ~500 ms).
    - **Unknown job_type** → parallel fan-out across Lambda, SageMaker, Step Functions;
      Batch is checked only when no other family returns hits.

    The primary hit is enriched (state machine tree, pipeline step logs, etc.).
    """
    return discover_compute_impl(
        query=query,
        job_type=job_type,
        max_results=max_results,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------

@app.get("/search/lambda", tags=["lambda"], summary="Search Lambda functions")
def search_lambda(
    query: Annotated[str | None, Query(description="Substring matched against name, description, runtime")] = None,
    name_prefix: Annotated[str | None, Query(description="Hard prefix filter on FunctionName")] = None,
    max_results: Annotated[int, Query(ge=1, le=100)] = 25,
    force_refresh: Annotated[bool, Query(description="Re-fetch function list from AWS")] = False,
) -> dict[str, Any]:
    """List Lambda functions matching the query.

    Results include `log_group` (`/aws/lambda/{name}`) — use it to pull
    CloudWatch Logs without any extra API call.

    The full function list is cached in-process and refreshed every ~4 minutes
    in the background. Pass `force_refresh=true` after a new deploy.
    """
    return search_lambda_compute_impl(
        query=query,
        name_prefix=name_prefix,
        max_results=max_results,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# Batch — existence check
# ---------------------------------------------------------------------------

@app.get("/search/batch", tags=["batch"], summary="Search Batch job definitions (existence check)")
def search_batch(
    query: Annotated[str | None, Query(description="Substring matched against job definition name, image, tags")] = None,
    status: Annotated[str, Query(description="Job definition status filter")] = "ACTIVE",
    max_results: Annotated[int, Query(ge=1, le=100)] = 25,
    include_queues: Annotated[bool, Query(description="Include matching job queues in the response")] = True,
) -> dict[str, Any]:
    """Check whether a Batch job definition exists.

    Pass 1 fans out `list_jobs(RUNNING)` across all enabled queues to catch
    actively-running jobs.  Pass 2 falls back to an exact + bounded page scan
    of job definitions.

    **For run history** (SUCCEEDED / FAILED), use `GET /search/batch/runs`.
    """
    return search_batch_compute_impl(
        query=query,
        status=status,
        max_results=max_results,
        include_queues=include_queues,
    )


# ---------------------------------------------------------------------------
# Batch — run history
# ---------------------------------------------------------------------------

@app.get("/search/batch/runs", tags=["batch"], summary="Search recent Batch job runs")
def search_batch_runs(
    query: Annotated[str | None, Query(description="Substring / wildcard matched against job name")] = None,
    statuses: Annotated[
        list[str] | None,
        Query(description="Run statuses to include. Default: SUCCEEDED, FAILED, RUNNING"),
    ] = None,
    max_results: Annotated[int, Query(ge=1, le=200)] = 20,
    max_per_queue: Annotated[
        int,
        Query(ge=1, le=100, description="Max results per (queue, status) pair — raise to dig deeper into history"),
    ] = 10,
    force_refresh: Annotated[bool, Query(description="Re-fetch queue list from AWS")] = False,
) -> dict[str, Any]:
    """Return recent job run history across all enabled Batch queues.

    Each result includes `job_id`, `queue`, `status`, `log_stream_name`,
    and `log_group` (`/aws/batch/job`).

    The queue list is cached; fan-out runs with 32 parallel workers.
    Typical latency: 1–3 s depending on the number of statuses requested.
    """
    return search_batch_runs_impl(
        query=query,
        statuses=statuses,
        max_results=max_results,
        max_per_queue=max_per_queue,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# Step Functions
# ---------------------------------------------------------------------------

@app.get("/search/stepfunctions", tags=["stepfunctions"], summary="Search Step Functions state machines")
def search_stepfunctions(
    query: Annotated[str | None, Query(description="Substring matched against name, ARN, type")] = None,
    max_results: Annotated[int, Query(ge=1, le=100)] = 25,
    state_machine_type: Annotated[
        str | None,
        Query(description="Filter by type: STANDARD or EXPRESS"),
    ] = None,
    force_refresh: Annotated[bool, Query(description="Re-fetch state machine list from AWS")] = False,
) -> dict[str, Any]:
    """Search for Step Functions state machines.

    Results are enriched with `logging_configuration`, `log_group` (extracted
    from the configured CloudWatch destination), and an ASL resource tree that
    shows every Lambda / Batch / nested SFN leaf.

    The full state machine list is cached in-process.
    """
    return search_stepfunctions_compute_impl(
        query=query,
        max_results=max_results,
        state_machine_type=state_machine_type,
        force_refresh=force_refresh,
    )


@app.get("/describe/state-machine-tree", tags=["stepfunctions"], summary="Walk the ASL resource tree of a state machine")
def describe_state_machine_tree(
    state_machine_arn: Annotated[str | None, Query(description="Full ARN of the state machine")] = None,
    state_machine_name: Annotated[str | None, Query(description="Name (resolved to ARN automatically)")] = None,
    max_depth: Annotated[int, Query(ge=1, le=10)] = 5,
) -> dict[str, Any]:
    """Walk an ASL definition and return every Lambda / Batch / nested SFN leaf.

    Useful after `/search/stepfunctions` returns a primary hit — pass the ARN
    here to see the full execution graph including Map and Parallel branches.
    """
    return describe_state_machine_tree_impl(
        state_machine_arn=state_machine_arn,
        state_machine_name=state_machine_name,
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# SageMaker
# ---------------------------------------------------------------------------

@app.get("/search/sagemaker", tags=["sagemaker"], summary="Search SageMaker pipelines, training jobs, and HPO")
def search_sagemaker(
    query: Annotated[str | None, Query(description="Substring matched against name, status, ARN")] = None,
    kind: Annotated[
        str | None,
        Query(description="Narrow to one surface: pipeline | training | hyperparameter"),
    ] = None,
    max_results: Annotated[int, Query(ge=1, le=100)] = 25,
    status_equals: Annotated[str | None, Query(description="Status filter for training/HPO (e.g. Completed)")] = None,
) -> dict[str, Any]:
    """Search SageMaker pipelines, training jobs, and hyperparameter tuning jobs.

    Pipeline results are enriched with `latest_execution_steps` — each step
    includes `log_group`, `job_type` (ProcessingJob / TrainingJob / …), and
    `job_name` so you can navigate directly to CloudWatch Logs.

    Training and HPO results include `log_group` (`/aws/sagemaker/TrainingJobs`).
    """
    return search_sagemaker_compute_impl(
        query=query,
        kind=kind,  # type: ignore[arg-type]
        max_results=max_results,
        status_equals=status_equals,
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@app.post("/cache/invalidate", tags=["cache"], summary="Invalidate the in-process resource cache")
def cache_invalidate(
    resource: Annotated[
        str,
        Query(description="Which cache to evict: sfn | lambda | batch_queues | all"),
    ] = "all",
) -> dict[str, Any]:
    """Evict one or all entries from the in-process resource cache.

    The cache holds Lambda function lists, SFN state machine lists, and Batch
    queue lists.  The next search will re-fetch from AWS synchronously and
    warm the cache again.

    Returns a snapshot of the cache state after eviction.
    """
    fn = invalidate_resource_cache.func if hasattr(invalidate_resource_cache, "func") else invalidate_resource_cache
    return fn(resource=resource)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Montai compute-discovery FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    uvicorn.run(
        "montai_joblib_utils_ai_mcp.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
