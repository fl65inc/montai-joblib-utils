"""
Strands agent that discovers existing Montai compute via search_* tools.

Companion to on-demand-compute-ai-mcp (submit/route). This agent only searches —
it does not submit jobs. Completion notifications (EventBridge → SNS) are out of
scope for v1; scientists check status in chat.

Usage:
  uv run montai-joblib-utils-agent "Find batch jobs related to acn"
  uv run montai-joblib-utils-agent
"""

from __future__ import annotations

import sys
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel

from montai_joblib_utils_ai_mcp.tools import (
    describe_state_machine_tree,
    discover_compute,
    invalidate_resource_cache,
    search_batch_runs,
    search_compute,
)

_SOP = Path(__file__).resolve().parent / "sops" / "compute-discovery.md"
_SOP_TEXT = _SOP.read_text() if _SOP.exists() else ""

SYSTEM_PROMPT = f"""\
You are the Montai Joblib Utils Assistant — you help scientists and platform
engineers discover compute that already exists in AWS.

## Your job
Find existing Lambda / Batch / SageMaker / Step Functions resources that match
the user's workload. Recommend reuse before anyone builds new ones.

## Speed rule (critical)
Call **`discover_compute` once**. Do NOT call family search tools yourself.
`discover_compute` already:
  1. Asks "do we know the job_type?"
  2. If yes → targeted search
  3. If no  → parallel fan-out across families

Then narrate the classification + strategy from the tool result, and report
`primary` (plus related hits).

Only use `search_compute` when the user already gave an explicit job_type and
you need a second targeted lookup.

## Cache behaviour
SFN, Lambda, and Batch queue lists are cached in-process (default TTL 5 min).
- Pass `force_refresh=True` to `discover_compute` or `search_batch_runs` to
  bypass cache in one call.
- Call `invalidate_resource_cache` (resource="sfn"|"lambda"|"batch_queues"|"all")
  when the user says something was recently deployed or asks for fresh AWS state.

## When to use search_batch_runs
Use `search_batch_runs` (not `discover_compute`) when the user explicitly asks
about **run history** — e.g. "show me recent runs of acn_predictions", "did this
job succeed?", "when did murcko_scaffolds last run?".
- Default statuses: SUCCEEDED, FAILED, RUNNING
- Returns actual job IDs, queue names, and timestamps
- `max_per_queue` (default 10) controls history depth — raise it to dig further back
- `discover_compute` is for existence checks; `search_batch_runs` is for execution history

## Guiding principles
- Deterministic search first: never invent ARNs.
- AWS is the source of truth: quote tool results.
- You do not submit jobs. Point at on-demand-compute-ai-mcp for that.
- v1 status check = in chat (no email / SNS completion flow yet).

## job_type menu
| job_type                   | Backend          |
|----------------------------|------------------|
| lambda                     | Lambda           |
| batch                      | Batch            |
| sagemaker_pipeline         | SageMaker        |
| sagemaker_training         | SageMaker        |
| sagemaker_hyperparameter   | SageMaker        |
| stepfunctions              | Step Functions   |
| state_machine              | Step Functions   |

## SOP (compute-discovery)
{_SOP_TEXT}

Keep answers short: name, job_type, key resources, ARN. Mention elapsed_ms.
"""

TOOLS = [
    discover_compute,
    describe_state_machine_tree,
    invalidate_resource_cache,
    search_batch_runs,
    search_compute,
]


def build_agent() -> Agent:
    """Construct the joblib-utils Strands agent."""
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name="us-east-1",
    )
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
    )


def main() -> None:
    agent = build_agent()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        agent(prompt)
    else:
        print("Montai Joblib Utils Agent — type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            agent(user_input)


if __name__ == "__main__":
    main()
