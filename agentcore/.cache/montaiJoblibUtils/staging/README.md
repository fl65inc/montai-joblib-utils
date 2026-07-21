# montai-joblib-utils-ai-mcp

Separate **test** library: Strands agent + FastMCP server that **searches** existing
Montai AWS compute (Lambda / Batch / SageMaker / Step Functions). Does not submit jobs.

Related: [PE-4718](https://montai.atlassian.net/browse/PE-4718) (job library submit
abstraction) and sibling package `on-demand-compute-ai-mcp` (route / submit).

v1 completion UX = check status in chat. EventBridge → SNS email is deferred.

## Layout

| Layer | Role |
|---|---|
| `*_impl` functions | Deterministic boto3 search |
| `@tool` wrappers | Agent-callable surface |
| `search_compute(job_type=...)` | Deterministic dispatcher |
| Strands `Agent` | Agentic orchestration over the tools |
| FastMCP server | Same tools over stdio / SSE |

## Tools

| Tool | Description |
|---|---|
| `discover_compute` | **Preferred.** Classify job_type → targeted or parallel fan-out |
| `search_compute` | Wrapper by known `job_type` |
| `search_lambda_compute` | List/filter Lambda functions |
| `search_batch_compute` | List/filter Batch job definitions (+ queues) |
| `search_sagemaker_compute` | Pipelines / training / HPO (`kind=`) |
| `search_stepfunctions_compute` | Step Functions state machines (`state_machine` alias) |

SOP: `src/montai_joblib_utils_ai_mcp/sops/compute-discovery.md`

### `job_type` values

`lambda` · `batch` · `sagemaker_pipeline` · `sagemaker_training` · `sagemaker_hyperparameter` · `stepfunctions` · `state_machine`

## Ideas from `platform_cdk/stacks/assets` + matrix stack

Existing compute often shows up as named Batch job defs / Glue-era scripts:

- `acn_predictions`, ADMET ETL scripts under `assets/`
- Matrix Batch defs: `murcko_scaffolds`, `np_classifier_predict`, …
- Platform Lambdas: CDD / inventory / vendor loaders, email forwarder

Search queries like `"acn"` or `"ondemand"` should surface those names from AWS.

## Quick start

```bash
cd montai-joblib-utils-ai-mcp
uv sync

# MCP (stdio — Cursor)
uv run montai-joblib-utils-mcp

# Agent REPL / one-shot
uv run montai-joblib-utils-agent "Find batch compute related to acn"
uv run montai-joblib-utils-agent
```

### Cursor MCP config

```json
{
  "mcpServers": {
    "montai-joblib-utils": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/montai-joblib-utils-ai-mcp", "montai-joblib-utils-mcp"],
      "env": {
        "AWS_PROFILE": "your-aws-profile"
      }
    }
  }
}
```

## Deterministic vs agentic

```python
from montai_joblib_utils_ai_mcp.tools.search_compute import search_compute_impl

# Pure function — no LLM
search_compute_impl(job_type="batch", query="acn")
```

```python
from montai_joblib_utils_ai_mcp.agent import build_agent

agent = build_agent()
agent("What Lambda runners do we already have for CDD?")
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/
```
