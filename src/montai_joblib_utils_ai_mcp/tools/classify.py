"""Deterministic job_type classifier — 'do we know what kind of job this is?'"""

from __future__ import annotations

import re
from typing import Any

from montai_joblib_utils_ai_mcp.types import JobType


def classify_job_type(
    query: str | None,
    job_type: JobType | None = None,
) -> dict[str, Any]:
    """Decide whether job_type is known before searching.

    Returns:
        {
          known: bool,
          job_type: JobType | None,
          confidence: "high" | "medium" | "low",
          reason: str,
        }
    """
    if job_type:
        return {
            "known": True,
            "job_type": job_type,
            "confidence": "high",
            "reason": f"Caller provided job_type={job_type!r}",
        }

    text = (query or "").strip().lower()
    if not text:
        return {
            "known": False,
            "job_type": None,
            "confidence": "low",
            "reason": "Empty query — cannot classify",
        }

    # High-confidence lexical cues (order matters — more specific first).
    rules: list[tuple[str, JobType, str]] = [
        (r"\bhyper[\s-]?parameter|\bhpo\b", "sagemaker_hyperparameter", "HPO / hyperparameter cue"),
        (r"\btraining\s+job\b|\bsagemaker\s+training\b", "sagemaker_training", "SageMaker training cue"),
        (
            r"\bsagemaker\s+pipeline\b|\bsm\s+pipeline\b|\binference\s+pipeline\b",
            "sagemaker_pipeline",
            "SageMaker pipeline cue",
        ),
        (
            r"\bstate\s*machines?\b|\bstep\s*functions?\b|\bsfn\b|\bfan[\s-]?outs?\b|\borchestration\b",
            "stepfunctions",
            "Step Functions / state machine cue",
        ),
        (r"\bbatch\s+job\b|\bjob\s+definitions?\b|\bjob\s+defs?\b|\bjob\s+queue\b", "batch", "Batch cue"),
        (r"\blambda\b|\bλ\b", "lambda", "Lambda cue"),
    ]

    for pattern, jt, reason in rules:
        if re.search(pattern, text):
            return {
                "known": True,
                "job_type": jt,
                "confidence": "high",
                "reason": reason,
            }

    # Medium: bare "pipeline" usually means SageMaker at Montai.
    if re.search(r"\bpipeline\b", text):
        return {
            "known": True,
            "job_type": "sagemaker_pipeline",
            "confidence": "medium",
            "reason": "Bare 'pipeline' cue → assume sagemaker_pipeline",
        }

    return {
        "known": False,
        "job_type": None,
        "confidence": "low",
        "reason": "No job_type cue in query — will fan out across families",
    }
