"""Extract distinctive search tokens from a natural-language workload query."""

from __future__ import annotations

import re

# Words that help classify job_type but poison substring match against AWS names.
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "me",
        "my",
        "on",
        "for",
        "of",
        "to",
        "in",
        "and",
        "or",
        "find",
        "info",
        "information",
        "about",
        "what",
        "which",
        "job",
        "jobs",
        "type",
        "compute",
        "runner",
        "run",
        "runs",
        "lambda",
        "batch",
        "sagemaker",
        "pipeline",
        "pipelines",
        "training",
        "hyperparameter",
        "hpo",
        "step",
        "steps",
        "function",
        "functions",
        "state",
        "machine",
        "machines",
        "sfn",
        "fan",
        "out",
        "fanout",
        "orchestration",
        "inference",
        "definition",
        "definitions",
        "queue",
        "queues",
        "sm",
    }
)


def extract_search_query(query: str | None) -> str:
    """Return the best substring to match against AWS resource names.

    Example: "tractability inference pipeline" → "tractability"
    """
    if not query or not query.strip():
        return ""
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", query.lower())
    kept = [t for t in tokens if t not in _STOP and len(t) >= 3]
    if not kept:
        return query.strip()
    # Prefer the longest distinctive token (usually the workload name).
    return max(kept, key=len)
