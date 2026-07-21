"""
montai-joblib-utils FastMCP server.

Exposes search_* tools for discovering existing AWS compute.
Does not submit jobs — pair with on-demand-compute-ai-mcp for that.

Usage:
  uv run montai-joblib-utils-mcp
  uv run montai-joblib-utils-mcp --sse --port 8081
"""

from __future__ import annotations

import argparse
import logging

from mcp.server.fastmcp import FastMCP

from montai_joblib_utils_ai_mcp.tools.discover_compute import discover_compute
from montai_joblib_utils_ai_mcp.tools.search_batch import search_batch_compute
from montai_joblib_utils_ai_mcp.tools.search_compute import search_compute
from montai_joblib_utils_ai_mcp.tools.search_lambda import search_lambda_compute
from montai_joblib_utils_ai_mcp.tools.search_sagemaker import search_sagemaker_compute
from montai_joblib_utils_ai_mcp.tools.search_stepfunctions import (
    search_stepfunctions_compute,
)
from montai_joblib_utils_ai_mcp.tools.state_machine_tree import (
    describe_state_machine_tree,
)

logging.basicConfig(level=logging.INFO)


def _unwrap(tool_obj):
    """Register the underlying callable whether or not strands @tool wrapped it."""
    return tool_obj.func if hasattr(tool_obj, "func") else tool_obj


mcp = FastMCP(
    "montai-joblib-utils",
    instructions=(
        "Discover existing Montai AWS compute. "
        "Prefer discover_compute(query=...) — it classifies job_type then does "
        "a targeted search or a parallel fan-out. "
        "Use search_compute(job_type=...) only when the type is already known. "
        "Do not submit jobs from this server."
    ),
)

mcp.tool()(_unwrap(discover_compute))
mcp.tool()(_unwrap(describe_state_machine_tree))
mcp.tool()(_unwrap(search_compute))
mcp.tool()(_unwrap(search_lambda_compute))
mcp.tool()(_unwrap(search_batch_compute))
mcp.tool()(_unwrap(search_sagemaker_compute))
mcp.tool()(_unwrap(search_stepfunctions_compute))


def main() -> None:
    parser = argparse.ArgumentParser(description="montai-joblib-utils MCP server")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run with SSE/HTTP transport. Default is stdio.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE mode.")
    parser.add_argument("--port", type=int, default=8081, help="Port for SSE mode.")
    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
