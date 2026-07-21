"""Search existing AWS Lambda functions (deterministic + Strands @tool)."""

from __future__ import annotations

import logging
from typing import Any

import boto3
from strands import tool

from montai_joblib_utils_ai_mcp.tools import _resource_cache as cache
from montai_joblib_utils_ai_mcp.tools._match import matches_query

logger = logging.getLogger(__name__)

_CACHE_KEY = "lambda"


def _list_all_functions(client: Any) -> list[dict[str, Any]]:
    """Paginate the full Lambda function list from AWS."""
    all_fns: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_functions")
    for page in paginator.paginate():
        all_fns.extend(page.get("Functions", []))
    logger.debug("Fetched %d Lambda functions from AWS", len(all_fns))
    return all_fns


def search_lambda_compute_impl(
    query: str | None = None,
    name_prefix: str | None = None,
    max_results: int = 25,
    region_name: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """List Lambda functions whose name/description/runtime match *query*.

    Inspired by Montai platform Lambdas in platform_cdk (DockerImageFunction
    inventory sync, CDD loaders, email forwarder, etc.) — the agent discovers
    what already exists before inventing a new runner.

    The full function list is cached in-process (TTL controlled by
    ``MONTAI_RESOURCE_CACHE_TTL``, default 300 s). Pass ``force_refresh=True``
    to bypass the cache and re-fetch from AWS, or call the
    ``invalidate_resource_cache`` tool.
    """
    client = boto3.client("lambda", region_name=region_name)
    matches: list[dict[str, Any]] = []

    try:
        all_fns = cache.get(_CACHE_KEY, region_name, force_refresh=force_refresh)
        if all_fns is None:
            all_fns = _list_all_functions(client)
            cache.set(
                _CACHE_KEY,
                region_name,
                all_fns,
                refresh_fn=lambda: _list_all_functions(
                    boto3.client("lambda", region_name=region_name)
                ),
            )

        for fn in all_fns:
            name = fn.get("FunctionName", "")
            if name_prefix and not name.startswith(name_prefix):
                continue
            desc = fn.get("Description") or ""
            runtime = fn.get("Runtime") or ""
            package_type = fn.get("PackageType") or ""
            if not matches_query(query, name, desc, runtime, package_type):
                continue
            matches.append(
                {
                    "name": name,
                    "arn": fn.get("FunctionArn"),
                    "runtime": runtime or None,
                    "package_type": package_type,
                    "timeout": fn.get("Timeout"),
                    "memory_mb": fn.get("MemorySize"),
                    "last_modified": fn.get("LastModified"),
                    "description": desc or None,
                    "log_group": f"/aws/lambda/{name}",
                    "job_type": "lambda",
                }
            )
            if len(matches) >= max_results:
                break
    except Exception as exc:  # noqa: BLE001
        logger.error("Lambda search failed: %s", exc)
        return {
            "job_type": "lambda",
            "query": query,
            "count": 0,
            "results": [],
            "error": str(exc),
        }

    return {
        "job_type": "lambda",
        "query": query,
        "name_prefix": name_prefix,
        "count": len(matches),
        "results": matches,
    }


@tool
def search_lambda_compute(
    query: str | None = None,
    name_prefix: str | None = None,
    max_results: int = 25,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Search AWS Lambda functions available for short synchronous compute.

    Use when looking for existing Lambdas (e.g. inventory sync, CDD loaders,
    prediction runners) before submitting a new on-demand job.

    The full function list is cached in-process and reused across concurrent
    searches. Pass ``force_refresh=True`` to bypass the cache and re-fetch the
    current list from AWS (e.g. after a new deploy).

    Args:
        query: Case-insensitive substring matched against name, description, runtime.
        name_prefix: Optional hard filter on FunctionName prefix.
        max_results: Cap on returned functions (default 25).
        force_refresh: When True, discard cached list and re-fetch from AWS.

    Returns:
        dict with job_type="lambda", count, and results list of function summaries.
    """
    return search_lambda_compute_impl(
        query=query,
        name_prefix=name_prefix,
        max_results=max_results,
        force_refresh=force_refresh,
    )
