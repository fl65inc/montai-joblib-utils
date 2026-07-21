"""Deterministic dispatcher: job_type → search_* backend."""

from __future__ import annotations

from typing import Any

from strands import tool

from montai_joblib_utils_ai_mcp.tools.search_batch import search_batch_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_lambda import search_lambda_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_sagemaker import search_sagemaker_compute_impl
from montai_joblib_utils_ai_mcp.tools.search_stepfunctions import (
    search_stepfunctions_compute_impl,
)
from montai_joblib_utils_ai_mcp.types import JOB_TYPES, JobType


def search_compute_impl(
    job_type: JobType,
    query: str | None = None,
    max_results: int = 25,
    name_prefix: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Route a compute search to the right AWS API based on *job_type*.

    job_type values:
      * lambda
      * batch
      * sagemaker_pipeline
      * sagemaker_training
      * sagemaker_hyperparameter
      * stepfunctions | state_machine
    """
    if job_type not in JOB_TYPES:
        return {
            "error": f"Unknown job_type={job_type!r}. Expected one of {list(JOB_TYPES)}",
            "job_type": job_type,
            "count": 0,
            "results": [],
        }

    if job_type == "lambda":
        return search_lambda_compute_impl(
            query=query,
            name_prefix=name_prefix,
            max_results=max_results,
        )

    if job_type == "batch":
        return search_batch_compute_impl(
            query=query,
            status=status or "ACTIVE",
            max_results=max_results,
        )

    if job_type in ("stepfunctions", "state_machine"):
        return search_stepfunctions_compute_impl(
            query=query,
            max_results=max_results,
            state_machine_type=status,  # optional STANDARD | EXPRESS
        )

    # sagemaker_* variants
    kind = job_type.removeprefix("sagemaker_")  # pipeline | training | hyperparameter
    return search_sagemaker_compute_impl(
        query=query,
        kind=kind,  # type: ignore[arg-type]
        max_results=max_results,
        status_equals=status,
    )


@tool
def search_compute(
    job_type: JobType,
    query: str | None = None,
    max_results: int = 25,
    name_prefix: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Search existing Montai compute by job_type (wrapper over tier-specific searchers).

    Prefer this when the caller already knows the compute family. Prefer the
    specific tools (search_lambda_compute / search_batch_compute /
    search_sagemaker_compute / search_stepfunctions_compute) when exploring.

    Args:
        job_type: One of lambda | batch | sagemaker_pipeline |
                  sagemaker_training | sagemaker_hyperparameter |
                  stepfunctions | state_machine.
        query: Case-insensitive substring filter on names / images / tags.
        max_results: Cap on returned items (default 25).
        name_prefix: Lambda-only hard prefix filter.
        status: Batch job-definition status (default ACTIVE), SageMaker
                training/HPO StatusEquals, or Step Functions type
                (STANDARD | EXPRESS) when job_type is stepfunctions.

    Returns:
        dict from the underlying search_* implementation.
    """
    return search_compute_impl(
        job_type=job_type,
        query=query,
        max_results=max_results,
        name_prefix=name_prefix,
        status=status,
    )
