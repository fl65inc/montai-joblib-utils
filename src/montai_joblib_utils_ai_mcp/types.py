"""Shared types for compute search."""

from __future__ import annotations

from typing import Literal

JobType = Literal[
    "lambda",
    "batch",
    "sagemaker_pipeline",
    "sagemaker_training",
    "sagemaker_hyperparameter",
    "stepfunctions",
    "state_machine",  # alias → stepfunctions backend
]

JOB_TYPES: tuple[JobType, ...] = (
    "lambda",
    "batch",
    "sagemaker_pipeline",
    "sagemaker_training",
    "sagemaker_hyperparameter",
    "stepfunctions",
    "state_machine",
)

# Map agent-facing job_type → which search backend to call
JOB_TYPE_BACKEND: dict[JobType, Literal["lambda", "batch", "sagemaker", "stepfunctions"]] = {
    "lambda": "lambda",
    "batch": "batch",
    "sagemaker_pipeline": "sagemaker",
    "sagemaker_training": "sagemaker",
    "sagemaker_hyperparameter": "sagemaker",
    "stepfunctions": "stepfunctions",
    "state_machine": "stepfunctions",
}
