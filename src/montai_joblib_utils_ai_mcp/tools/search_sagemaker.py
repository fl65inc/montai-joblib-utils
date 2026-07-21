"""Search SageMaker pipelines, training jobs, and HPO jobs."""

from __future__ import annotations

import logging
from typing import Any, Literal

import boto3
from strands import tool

from montai_joblib_utils_ai_mcp.tools._match import matches_query

logger = logging.getLogger(__name__)

SageMakerKind = Literal["pipeline", "training", "hyperparameter"]

# Log group for each step type that appears in pipeline execution step Metadata.
_STEP_TYPE_LOG_GROUP: dict[str, str] = {
    "ProcessingJob": "/aws/sagemaker/ProcessingJobs",
    "TrainingJob": "/aws/sagemaker/TrainingJobs",
    "TransformJob": "/aws/sagemaker/TransformJobs",
}


def _job_name_from_arn(arn: str) -> str:
    """Return the resource name (last segment) of a SageMaker ARN."""
    return arn.rsplit("/", 1)[-1] if arn else ""


def _latest_execution_steps(client: Any, pipeline_name: str) -> list[dict[str, Any]]:
    """Return enriched step summaries from the most recent pipeline execution.

    Each step entry includes:
      step_name, step_status, job_type (ProcessingJob/TrainingJob/…),
      job_name, job_arn, log_group.
    Returns [] if no executions exist or any API call fails.
    """
    try:
        execs = client.list_pipeline_executions(
            PipelineName=pipeline_name,
            MaxResults=1,
            SortBy="CreationTime",
            SortOrder="Descending",
        ).get("PipelineExecutionSummaries", [])
        if not execs:
            return []

        exec_arn = execs[0]["PipelineExecutionArn"]
        raw_steps = client.list_pipeline_execution_steps(
            PipelineExecutionArn=exec_arn,
            MaxResults=30,
        ).get("PipelineExecutionSteps", [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("pipeline execution steps fetch failed for %s: %s", pipeline_name, exc)
        return []

    steps = []
    for s in raw_steps:
        meta = s.get("Metadata") or {}
        # Find the first recognised step type in the metadata dict.
        job_type = next((k for k in _STEP_TYPE_LOG_GROUP if k in meta), None)
        job_arn: str | None = None
        job_name: str | None = None
        log_group: str | None = None

        if job_type:
            job_arn = (meta[job_type] or {}).get("Arn")
            job_name = _job_name_from_arn(job_arn or "")
            log_group = _STEP_TYPE_LOG_GROUP[job_type]
        elif "Lambda" in meta:
            # Lambda step — log group is deterministic from function name/ARN
            lambda_arn = (meta["Lambda"] or {}).get("Arn") or ""
            job_type = "Lambda"
            job_arn = lambda_arn
            job_name = _job_name_from_arn(lambda_arn)
            log_group = f"/aws/lambda/{job_name}" if job_name else None

        steps.append({
            "step_name": s.get("StepName"),
            "step_status": s.get("StepStatus"),
            "job_type": job_type,
            "job_name": job_name or None,
            "job_arn": job_arn or None,
            "log_group": log_group,
        })
    return steps


def _enrich_pipeline(client: Any, name: str, base: dict[str, Any]) -> dict[str, Any]:
    """Add describe_pipeline fields and latest-execution step log info."""
    try:
        detail = client.describe_pipeline(PipelineName=name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("describe_pipeline(%s) failed: %s", name, exc)
        return base
    definition = detail.get("PipelineDefinition") or ""
    if len(definition) > 2000:
        definition = definition[:2000] + "…[truncated]"

    steps = _latest_execution_steps(client, name)

    base.update(
        {
            "role_arn": detail.get("RoleArn"),
            "pipeline_status": detail.get("PipelineStatus"),
            "description": detail.get("PipelineDescription") or None,
            "parallelism_config": detail.get("ParallelismConfiguration"),
            "definition_preview": definition or None,
            "latest_execution_steps": steps or None,
        }
    )
    return base


def _search_pipelines(
    client: Any,
    query: str | None,
    max_results: int,
    enrich: bool = True,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_pipelines")
    for page in paginator.paginate():
        for p in page.get("PipelineSummaries", []):
            name = p.get("PipelineName", "")
            display = p.get("PipelineDisplayName") or ""
            if not matches_query(query, name, display, p.get("PipelineArn")):
                continue
            summary = {
                "name": name,
                "arn": p.get("PipelineArn"),
                "display_name": display or None,
                "creation_time": str(p.get("CreationTime") or ""),
                "last_modified": str(p.get("LastModifiedTime") or ""),
                "job_type": "sagemaker_pipeline",
            }
            if enrich:
                summary = _enrich_pipeline(client, name, summary)
            matches.append(summary)
            if len(matches) >= max_results:
                return matches
    return matches


def _search_training(
    client: Any,
    query: str | None,
    max_results: int,
    status_equals: str | None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {"MaxResults": min(max_results, 100), "SortBy": "CreationTime", "SortOrder": "Descending"}
    if status_equals:
        kwargs["StatusEquals"] = status_equals
    # list_training_jobs is not paginated the same way across all SDK versions;
    # use NextToken loop for robustness.
    next_token: str | None = None
    while len(matches) < max_results:
        if next_token:
            kwargs["NextToken"] = next_token
        resp = client.list_training_jobs(**kwargs)
        for job in resp.get("TrainingJobSummaries", []):
            name = job.get("TrainingJobName", "")
            status = job.get("TrainingJobStatus", "")
            if not matches_query(query, name, status, job.get("TrainingJobArn")):
                continue
            matches.append(
                {
                    "name": name,
                    "arn": job.get("TrainingJobArn"),
                    "status": status,
                    "creation_time": str(job.get("CreationTime") or ""),
                    "training_end_time": str(job.get("TrainingEndTime") or ""),
                    "log_group": "/aws/sagemaker/TrainingJobs",
                    "job_type": "sagemaker_training",
                }
            )
            if len(matches) >= max_results:
                break
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return matches


def _search_hyperparameter(
    client: Any,
    query: str | None,
    max_results: int,
    status_equals: str | None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        "MaxResults": min(max_results, 100),
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
    }
    if status_equals:
        kwargs["StatusEquals"] = status_equals
    next_token: str | None = None
    while len(matches) < max_results:
        if next_token:
            kwargs["NextToken"] = next_token
        resp = client.list_hyper_parameter_tuning_jobs(**kwargs)
        for job in resp.get("HyperParameterTuningJobSummaries", []):
            name = job.get("HyperParameterTuningJobName", "")
            status = job.get("HyperParameterTuningJobStatus", "")
            if not matches_query(query, name, status, job.get("HyperParameterTuningJobArn")):
                continue
            matches.append(
                {
                    "name": name,
                    "arn": job.get("HyperParameterTuningJobArn"),
                    "status": status,
                    "creation_time": str(job.get("CreationTime") or ""),
                    "training_job_status_counters": job.get("TrainingJobStatusCounters"),
                    "log_group": "/aws/sagemaker/TrainingJobs",
                    "job_type": "sagemaker_hyperparameter",
                }
            )
            if len(matches) >= max_results:
                break
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return matches


def search_sagemaker_compute_impl(
    query: str | None = None,
    kind: SageMakerKind | None = None,
    max_results: int = 25,
    status_equals: str | None = None,
    region_name: str | None = None,
    enrich: bool = True,
) -> dict[str, Any]:
    """Search SageMaker pipelines and/or training / HPO jobs.

    ``kind`` narrows the surface:
      * pipeline — list_pipelines
      * training — list_training_jobs
      * hyperparameter — list_hyper_parameter_tuning_jobs
      * None — search all three (capped per kind by max_results)
    """
    client = boto3.client("sagemaker", region_name=region_name)
    results: list[dict[str, Any]] = []
    kinds: list[SageMakerKind] = [kind] if kind else ["pipeline", "training", "hyperparameter"]

    try:
        for k in kinds:
            if k == "pipeline":
                results.extend(_search_pipelines(client, query, max_results, enrich=enrich))
            elif k == "training":
                results.extend(_search_training(client, query, max_results, status_equals))
            elif k == "hyperparameter":
                results.extend(_search_hyperparameter(client, query, max_results, status_equals))
            if kind and len(results) >= max_results:
                break
        if kind:
            results = results[:max_results]
    except Exception as exc:  # noqa: BLE001
        logger.error("SageMaker search failed: %s", exc)
        return {
            "job_type": f"sagemaker_{kind}" if kind else "sagemaker",
            "query": query,
            "kind": kind,
            "count": 0,
            "results": [],
            "error": str(exc),
        }

    job_type = f"sagemaker_{kind}" if kind else "sagemaker"
    return {
        "job_type": job_type,
        "query": query,
        "kind": kind,
        "count": len(results),
        "results": results,
    }


@tool
def search_sagemaker_compute(
    query: str | None = None,
    kind: SageMakerKind | None = None,
    max_results: int = 25,
    status_equals: str | None = None,
) -> dict[str, Any]:
    """Search SageMaker pipelines, training jobs, and/or hyperparameter tuning jobs.

    Args:
        query: Case-insensitive substring matched against name / status / ARN.
        kind: Optional narrow filter — "pipeline" | "training" | "hyperparameter".
              Omit to search all three surfaces.
        max_results: Cap per kind when kind is set; overall soft cap when omitted.
        status_equals: Optional status filter for training / HPO (e.g. Completed).

    Returns:
        dict with job_type, kind, count, and results list.
    """
    return search_sagemaker_compute_impl(
        query=query,
        kind=kind,
        max_results=max_results,
        status_equals=status_equals,
    )
