"""Search AWS Step Functions state machines (deterministic + Strands @tool)."""

from __future__ import annotations

import logging
from typing import Any

import boto3
from strands import tool

from montai_joblib_utils_ai_mcp.tools import _resource_cache as cache
from montai_joblib_utils_ai_mcp.tools._match import matches_query
from montai_joblib_utils_ai_mcp.tools.state_machine_tree import attach_tree_to_summary

logger = logging.getLogger(__name__)

_CACHE_KEY = "sfn"


def _enrich_state_machine(
    client: Any,
    arn: str,
    base: dict[str, Any],
    *,
    include_tree: bool = True,
) -> dict[str, Any]:
    """Add describe_state_machine fields (+ optional ASL resource tree)."""
    try:
        detail = client.describe_state_machine(stateMachineArn=arn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("describe_state_machine(%s) failed: %s", arn, exc)
        return base

    logging_cfg = detail.get("loggingConfiguration") or {}
    # Extract the first CW log group ARN from loggingConfiguration.destinations
    log_group: str | None = None
    for dest in logging_cfg.get("destinations") or []:
        lg = (dest.get("cloudWatchLogsLogGroup") or {}).get("logGroupArn")
        if lg:
            # ARN format: arn:aws:logs:region:account:log-group:/name:*  — strip the trailing :*
            log_group = lg.rstrip(":*")
            break

    base.update(
        {
            "status": detail.get("status"),
            "role_arn": detail.get("roleArn"),
            "type": detail.get("type") or base.get("type"),
            "creation_date": str(detail.get("creationDate") or base.get("creation_date") or ""),
            "logging_configuration": logging_cfg or None,
            "log_group": log_group,
            "tracing_configuration": detail.get("tracingConfiguration"),
            "description": detail.get("description") or None,
        }
    )
    if include_tree:
        base = attach_tree_to_summary(client, base)
    else:
        definition = detail.get("definition") or ""
        if len(definition) > 2000:
            definition = definition[:2000] + "…[truncated]"
        base["definition_preview"] = definition or None
    return base


def _list_all_state_machines(client: Any) -> list[dict[str, Any]]:
    """Paginate the full SFN state machine list from AWS."""
    all_sms: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_state_machines")
    for page in paginator.paginate():
        all_sms.extend(page.get("stateMachines", []))
    logger.debug("Fetched %d state machines from AWS", len(all_sms))
    return all_sms


def search_stepfunctions_compute_impl(
    query: str | None = None,
    max_results: int = 25,
    state_machine_type: str | None = None,
    region_name: str | None = None,
    enrich: bool = True,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """List Step Functions state machines matching *query*.

    Covers STANDARD and EXPRESS state machines — the orchestration layer that
    often parents Lambda/Batch runs (Jobs API run-tree roots).

    The full state machine list is cached in-process (TTL controlled by
    ``MONTAI_RESOURCE_CACHE_TTL``, default 300 s). Pass ``force_refresh=True``
    to bypass the cache and re-fetch from AWS, or call the
    ``invalidate_resource_cache`` tool.
    """
    client = boto3.client("stepfunctions", region_name=region_name)
    matches: list[dict[str, Any]] = []

    try:
        all_sms = cache.get(_CACHE_KEY, region_name, force_refresh=force_refresh)
        if all_sms is None:
            all_sms = _list_all_state_machines(client)
            cache.set(
                _CACHE_KEY,
                region_name,
                all_sms,
                refresh_fn=lambda: _list_all_state_machines(
                    boto3.client("stepfunctions", region_name=region_name)
                ),
            )

        for sm in all_sms:
            name = sm.get("name", "")
            arn = sm.get("stateMachineArn", "")
            sm_type = sm.get("type", "")
            if state_machine_type and sm_type.upper() != state_machine_type.upper():
                continue
            if not matches_query(query, name, arn, sm_type):
                continue
            summary: dict[str, Any] = {
                "name": name,
                "arn": arn,
                "type": sm_type or None,
                "creation_date": str(sm.get("creationDate") or ""),
                "job_type": "stepfunctions",
            }
            if enrich and arn:
                summary = _enrich_state_machine(client, arn, summary)
            matches.append(summary)
            if len(matches) >= max_results:
                break
    except Exception as exc:  # noqa: BLE001
        logger.error("Step Functions search failed: %s", exc)
        return {
            "job_type": "stepfunctions",
            "query": query,
            "count": 0,
            "results": [],
            "error": str(exc),
        }

    return {
        "job_type": "stepfunctions",
        "query": query,
        "state_machine_type": state_machine_type,
        "count": len(matches),
        "results": matches,
    }


@tool
def search_stepfunctions_compute(
    query: str | None = None,
    max_results: int = 25,
    state_machine_type: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Search AWS Step Functions state machines (orchestration / run-tree roots).

    Use for fan-out workflows that call Lambda or Batch (e.g. tractability
    batch generators, matrix pipelines). "State machine" and "Step Functions"
    refer to the same AWS resource family.

    The full state machine list is cached in-process and reused across
    concurrent searches. Pass ``force_refresh=True`` to bypass the cache and
    re-fetch the current list from AWS (e.g. after a new deploy).

    Args:
        query: Case-insensitive substring matched against name / ARN / type.
        max_results: Cap on returned state machines (default 25).
        state_machine_type: Optional filter — "STANDARD" or "EXPRESS".
        force_refresh: When True, discard cached list and re-fetch from AWS.

    Returns:
        dict with job_type="stepfunctions", count, and results (enriched summaries).
    """
    return search_stepfunctions_compute_impl(
        query=query,
        max_results=max_results,
        state_machine_type=state_machine_type,
        force_refresh=force_refresh,
    )


# Alias for callers / docs that say "state machine"
search_state_machine_compute = search_stepfunctions_compute
search_state_machine_compute_impl = search_stepfunctions_compute_impl
