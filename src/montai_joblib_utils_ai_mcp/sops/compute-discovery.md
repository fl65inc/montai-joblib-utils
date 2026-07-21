# SOP: Compute Discovery

**Purpose:** Find existing Montai AWS compute for a named workload — fast.

**Principle:** SOPs hold the intelligence; search tools stay dumb. The agent
does not fan out tools itself.

## Decision tree

```
User asks about a workload
        │
        ▼
Do we know the job_type?
  (user said it OR classifier is high-confidence)
        │
   YES ─┴─► search_compute(job_type=…)     # one targeted AWS call
        │
   NO  ─┴─► staged fan-out
              Phase 1 (parallel, fast):
                lambda ‖ sagemaker_pipeline ‖ stepfunctions (state machines)
              If hits → done (skip Batch)
              Phase 2 (slow fallback):
                batch  (DescribeJobDefinitions is expensive)
              merge hits → pick primary → report
```

## Classifier cues (deterministic)

| Cue in query | job_type |
|---|---|
| state machine, step function(s), sfn, fan-out, fanout, orchestration | `stepfunctions` |
| sagemaker pipeline, sm pipeline, training job, hyperparameter, hpo | matching `sagemaker_*` |
| lambda, λ function (as compute) | `lambda` |
| batch job, job definition, job def, job queue | `batch` |
| "pipeline" alone (no sagemaker/step) | medium → `sagemaker_pipeline` |
| none of the above | unknown → parallel fan-out |

Explicit `job_type` from the caller always wins.

## Agent rules

1. Call **`discover_compute` once**. Do not call the family tools yourself.
2. Say out loud whether job_type was known or you are fanning out.
3. Report `primary` plus any related hits (helpers).
4. Never invent ARNs — only quote tool results.

## Cache behaviour

SFN list, Lambda list, and Batch queue list are cached in-process and refreshed
automatically every ~4 minutes by a background daemon thread.
**Default: always read from cache.**

| Situation | What to do |
|---|---|
| Normal existence check | `discover_compute(query=…)` — reads from cache, fast |
| Run history search | `search_batch_runs(query=…)` — see section below |
| User says something was just deployed / "check again" | `discover_compute(query=…, force_refresh=True)` |
| Fresh queue list for run history | `search_batch_runs(query=…, force_refresh=True)` |
| Pre-clear before a batch of searches | `invalidate_resource_cache(resource="all")` then search normally |
| User asks about cache state | `invalidate_resource_cache` returns `cache_after` snapshot |

`force_refresh=True` evicts the stale entry, fetches live from AWS once, stores
the result, and the background refresh thread continues on its normal schedule.

## Batch run history — search_batch_runs

Use **`search_batch_runs`** (not `discover_compute`) whenever the user asks about
**job execution history** rather than resource existence:

> "Show me recent runs of acn_predictions"
> "Did murcko_scaffolds succeed?"
> "When did this job last run?"
> "Are any property jobs currently running?"

| Parameter | Default | Notes |
|---|---|---|
| `query` | None | Substring / wildcard matched against job name |
| `statuses` | `["SUCCEEDED","FAILED","RUNNING"]` | Pass `["RUNNING"]` for active-only |
| `max_results` | 20 | Total across all queues |
| `max_per_queue` | 10 | Per (queue, status) pair — raise to dig deeper |
| `force_refresh` | False | Re-fetch queue list from AWS |

The Batch queue list is cached — the fan-out across 46 queues × statuses uses
cached queue names and runs with 32 parallel workers.  Typical latency: 1–3 s
depending on the number of statuses requested.

## State machine trees

When the primary hit is a Step Functions state machine, enrich with
`describe_state_machine_tree` — walk the ASL for Lambda / Batch / nested SFN
leaves (Map/Parallel included). Matrix properties almost always resolve to an
SFN that submits a Batch job def of the same name.

## Speed notes

- **Two Batch tools, two purposes.**
  `discover_compute` / `search_batch_compute` = existence check (is there a definition?).
  `search_batch_runs` = run history (did it execute, when, which queue, success/fail?).
- **Batch existence path:** RUNNING fan-out (46 calls) → exact definition lookup (1 call, O(1)) → bounded page scan (fallback only). Fast because definition deduplication skips all stale revisions (some definitions have thousands).
- **Batch run-history path:** `list_jobs` fanned out across 46 queues × statuses in parallel (32 workers). Queue list is cached. Typical latency 1–3 s.
- **Fan-out skips Batch entirely** when Phase 1 (Lambda / SageMaker / SFN) already found hits.
- `result_source` in `search_batch_compute` responses tells you whether hits came from runs or definitions.
- Enrichment (`describe_*` / tree walk) runs only on the primary hit.
- Targeted path is always preferred when classification is high/medium confidence.
