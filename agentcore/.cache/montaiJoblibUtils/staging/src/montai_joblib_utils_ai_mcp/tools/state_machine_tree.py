"""Walk a Step Functions ASL definition into a resource tree.

State machines can call Lambda, Batch, nested SFNs, Glue, SNS, etc.
This builds the Jobs-API-style parent/child view from the *definition*
(catalog tree), not from a live execution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from strands import tool

logger = logging.getLogger(__name__)

_MAX_DEPTH = 4


class _LazyClient:
    """Create a boto3 Step Functions client only if nested SFN recursion needs it."""

    def __init__(self, region_name: str | None = None) -> None:
        self._region = region_name or "us-east-1"
        self._client: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._client is None:
            self._client = boto3.client("stepfunctions", region_name=self._region)
        return getattr(self._client, name)


def _classify_resource(resource: str | None) -> str:
    if not resource:
        return "unknown"
    r = resource.lower()
    if "lambda" in r and ("function" in r or r.startswith("arn:aws:lambda:")):
        return "lambda"
    # Optimized Batch integration — not a job-queue ARN.
    if "batch:submitjob" in r or "states:::batch:" in r:
        return "batch"
    if "job-definition/" in r or r.startswith("arn:aws:batch:") and "job-definition" in r:
        return "batch"
    if "states:startexecution" in r or "states:::states:start" in r:
        return "stepfunctions"
    if "glue:startjobrun" in r:
        return "glue"
    if "sns:publish" in r:
        return "sns"
    if "sqs:sendmessage" in r:
        return "sqs"
    if "ecs:runtask" in r:
        return "ecs"
    if "sagemaker:" in r:
        return "sagemaker"
    if r.startswith("arn:aws:states:") and ":statemachine:" in r:
        return "stepfunctions"
    if r.startswith("arn:aws:lambda:"):
        return "lambda"
    return "task"


def _arn_tail(arn: str | None) -> str | None:
    if not arn or not isinstance(arn, str):
        return None
    return arn.split(":")[-1].split("/")[-1]


def _extract_targets(state: dict[str, Any], resource: str | None) -> list[dict[str, Any]]:
    """Pull JobDefinition / FunctionName / StateMachineArn out of Parameters."""
    params = state.get("Parameters") or {}
    targets: list[dict[str, Any]] = []
    kind = _classify_resource(resource)

    if kind == "batch":
        jd = params.get("JobDefinition")
        jq = params.get("JobQueue")
        if jd and isinstance(jd, str):
            targets.append(
                {
                    "job_type": "batch",
                    "name": _arn_tail(jd) if jd.startswith("arn:") else jd,
                    "arn": jd if jd.startswith("arn:") else None,
                    "job_queue": jq if isinstance(jq, str) else None,
                }
            )
    elif kind == "lambda":
        # Resource is often the full function ARN for optimized integrations.
        name = _arn_tail(resource)
        fn = params.get("FunctionName")
        if fn and isinstance(fn, str):
            name = _arn_tail(fn) or fn
            targets.append({"job_type": "lambda", "name": name, "arn": fn})
        elif name:
            targets.append(
                {
                    "job_type": "lambda",
                    "name": name,
                    "arn": resource if resource and resource.startswith("arn:") else None,
                }
            )
    elif kind == "stepfunctions":
        sm = params.get("StateMachineArn") or params.get("stateMachineArn")
        if sm and isinstance(sm, str):
            targets.append(
                {
                    "job_type": "stepfunctions",
                    "name": _arn_tail(sm),
                    "arn": sm,
                }
            )

    # Catch compute ARNs nested in Parameters — ignore queues / roles / noise.
    def _walk_strings(obj: Any) -> None:
        if isinstance(obj, str) and obj.startswith("arn:aws:"):
            low = obj.lower()
            if "job-queue/" in low or ":role/" in low or "event-bus" in low:
                return
            k = _classify_resource(obj)
            if k in ("lambda", "batch", "stepfunctions", "glue", "sagemaker"):
                entry = {"job_type": k, "name": _arn_tail(obj), "arn": obj}
                if not any(t.get("arn") == obj for t in targets):
                    targets.append(entry)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk_strings(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk_strings(v)

    _walk_strings(params)
    return targets


def _walk_states(
    states: dict[str, Any],
    *,
    client: Any,
    depth: int,
    seen: set[str],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for name, state in (states or {}).items():
        if not isinstance(state, dict):
            continue
        stype = state.get("Type", "Unknown")
        node: dict[str, Any] = {
            "state": name,
            "type": stype,
            "children": [],
        }

        if stype == "Task":
            resource = state.get("Resource")
            node["resource"] = resource
            node["resource_kind"] = _classify_resource(resource)
            node["targets"] = _extract_targets(state, resource)
            # Recurse into nested state machine ARNs.
            for t in node["targets"]:
                if t.get("job_type") == "stepfunctions" and t.get("arn") and depth < _MAX_DEPTH:
                    nested = describe_state_machine_tree_impl(
                        state_machine_arn=t["arn"],
                        client=client,
                        depth=depth + 1,
                        seen=seen,
                    )
                    if nested.get("tree"):
                        node["children"].append(nested["tree"])

        elif stype == "Map":
            iterator = state.get("Iterator") or state.get("ItemProcessor") or {}
            nested_states = iterator.get("States") if isinstance(iterator, dict) else None
            if nested_states:
                node["children"] = _walk_states(
                    nested_states, client=client, depth=depth, seen=seen
                )

        elif stype == "Parallel":
            for i, branch in enumerate(state.get("Branches") or []):
                branch_states = branch.get("States") if isinstance(branch, dict) else None
                if branch_states:
                    node["children"].append(
                        {
                            "state": f"Branch[{i}]",
                            "type": "Branch",
                            "children": _walk_states(
                                branch_states, client=client, depth=depth, seen=seen
                            ),
                        }
                    )

        nodes.append(node)
    return nodes


def _collect_leaves(nodes: list[dict[str, Any]], leaves: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Flatten target resources for quick agent consumption."""
    if leaves is None:
        leaves = []
    for n in nodes:
        for t in n.get("targets") or []:
            leaves.append(t)
        _collect_leaves(n.get("children") or [], leaves)
    return leaves


def describe_state_machine_tree_impl(
    *,
    state_machine_arn: str | None = None,
    state_machine_name: str | None = None,
    definition: str | dict | None = None,
    client: Any | None = None,
    depth: int = 0,
    seen: set[str] | None = None,
    region_name: str | None = None,
) -> dict[str, Any]:
    """Return the ASL resource tree for a state machine."""
    seen = seen if seen is not None else set()

    name = state_machine_name
    arn = state_machine_arn
    sm_type = None
    status = None

    def _client() -> Any:
        nonlocal client
        if client is None:
            client = boto3.client("stepfunctions", region_name=region_name or "us-east-1")
        return client

    if definition is None:
        client = _client()
        if not arn and name:
            # Resolve name → ARN via list (small account; fine for now).
            for page in client.get_paginator("list_state_machines").paginate():
                for sm in page.get("stateMachines", []):
                    if sm.get("name") == name or sm.get("name", "").lower() == name.lower():
                        arn = sm["stateMachineArn"]
                        break
                if arn:
                    break
        if not arn:
            return {
                "error": "Provide state_machine_arn, state_machine_name, or definition",
                "tree": None,
                "leaves": [],
            }
        if arn in seen:
            return {
                "name": name or _arn_tail(arn),
                "arn": arn,
                "tree": {"state": _arn_tail(arn), "type": "StateMachine", "cycle": True, "children": []},
                "leaves": [],
            }
        seen.add(arn)
        try:
            detail = client.describe_state_machine(stateMachineArn=arn)
        except Exception as exc:  # noqa: BLE001
            logger.error("describe_state_machine failed: %s", exc)
            return {"error": str(exc), "arn": arn, "tree": None, "leaves": []}
        name = detail.get("name") or name
        sm_type = detail.get("type")
        status = detail.get("status")
        definition = detail.get("definition") or "{}"

    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid ASL JSON: {exc}", "tree": None, "leaves": []}

    assert isinstance(definition, dict)
    start_at = definition.get("StartAt")
    states = definition.get("States") or {}
    # Lazy client: only needed if nested StartExecution targets appear.
    walk_client = client if client is not None else _LazyClient(region_name)
    children = _walk_states(states, client=walk_client, depth=depth, seen=seen)
    tree = {
        "state": name or _arn_tail(arn) or "StateMachine",
        "type": "StateMachine",
        "start_at": start_at,
        "job_type": "stepfunctions",
        "arn": arn,
        "children": children,
    }
    leaves = _collect_leaves(children)
    # Dedupe leaves by arn/name
    deduped: list[dict[str, Any]] = []
    seen_leaf: set[str] = set()
    for leaf in leaves:
        key = leaf.get("arn") or f"{leaf.get('job_type')}:{leaf.get('name')}"
        if key in seen_leaf:
            continue
        seen_leaf.add(key)
        deduped.append(leaf)

    return {
        "name": name,
        "arn": arn,
        "type": sm_type,
        "status": status,
        "job_type": "stepfunctions",
        "tree": tree,
        "leaves": deduped,
        "leaf_count": len(deduped),
        "depth": depth,
    }


def attach_tree_to_summary(client: Any, summary: dict[str, Any]) -> dict[str, Any]:
    """Add tree + leaves onto a search_stepfunctions result row."""
    arn = summary.get("arn")
    if not arn:
        return summary
    tree_out = describe_state_machine_tree_impl(state_machine_arn=arn, client=client)
    summary = dict(summary)
    summary["tree"] = tree_out.get("tree")
    summary["leaves"] = tree_out.get("leaves") or []
    summary["leaf_count"] = tree_out.get("leaf_count", 0)
    # Drop huge definition_preview once we have a structured tree.
    if summary.get("tree") and "definition_preview" in summary:
        summary.pop("definition_preview", None)
    return summary


@tool
def describe_state_machine_tree(
    state_machine_name: str | None = None,
    state_machine_arn: str | None = None,
) -> dict[str, Any]:
    """Walk a Step Functions state machine ASL into a full resource tree.

    Use after discover/search finds a state machine. Returns nested states plus
    leaf targets (Lambda, Batch job defs, nested SFNs, etc.).

    Args:
        state_machine_name: Exact or case-insensitive state machine name.
        state_machine_arn: Full state machine ARN (preferred if known).

    Returns:
        tree (nested states), leaves (flat resource list), leaf_count.
    """
    return describe_state_machine_tree_impl(
        state_machine_name=state_machine_name,
        state_machine_arn=state_machine_arn,
    )
