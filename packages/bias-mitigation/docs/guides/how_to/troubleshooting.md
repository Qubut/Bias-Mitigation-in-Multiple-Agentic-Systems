# Troubleshooting

This page lists common runtime and evaluation issues and their practical fixes.

## `mem0g` Run Hangs or Becomes Unresponsive

Symptoms:

- `uv run train.py --intervention mem0g` appears stuck after startup logs.
- Keyboard interrupt responsiveness is degraded.

Actions:

- Ensure telemetry is disabled for Mem0 paths (`MEM0_TELEMETRY=False`).
- Validate memory backend dependencies (vector store, optional graph store) are reachable.
- Run with minimal config and reduced sample size to isolate backend connectivity.

## `mem0g` Occupies All CPU Cores

Symptoms:

- `uv run evaluate ... --intervention mem0g` spikes to full-core CPU usage.
- Baseline intervention does not show the same saturation profile.

Actions:

- Use `evaluator_num_threads` in `configs/mas_config.yaml` to cap evaluation concurrency.
- For GenAI backend, runtime now maps that value to `MLFLOW_GENAI_EVAL_MAX_WORKERS` during evaluation.
- For `mem0g` interventions, runtime now applies a thread guard (when unset) for:
	`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`.
- Inspect logged concurrency snapshot in `evaluation/summary.json` and MLflow artifact `evaluation/concurrency_snapshot.json`.

## Agent State / Memory Lifecycle Mismatch

Symptoms:

- Agent performs interaction behavior during expected genesis step.
- Memory retrieval or persistence appears in unexpected phases.
- Runtime raises transition guard errors from `AgentStateMachine`.

Actions:

- Confirm `peer_answers` is only provided during interaction rounds.
- Verify `AgentStateMachine` transitions (`genesis -> interaction -> completed`) are not bypassed.
- Keep `reset_memory_on_genesis` disabled unless strict reset behavior is required and measured.

## Memory Clear Policy Causes Slowdowns

Symptoms:

- `mem0g` runs are responsive but significantly slower than baseline.
- Runtime logs show repeated memory clear activity at genesis.

Actions:

- Prefer `reset_memory_on_genesis: false` when session IDs are already run-scoped.
- Enable `reset_memory_on_genesis: true` only for strict isolation experiments where added cleanup overhead is acceptable.
- Check MLflow metrics `memory.clear.attempts` and `memory.clear.successes` to quantify cleanup impact.

## `mem0g` Evaluation Appears Stuck at `0%`

Symptoms:

- Deterministic evaluation starts but progress stays at `0%` for a long time.
- `Ctrl+C` feels delayed while memory and model operations are in flight.

Actions:

- Keep `evaluator_num_threads` moderate for `mem0g` (start with `2` to `4`).
- Tune `evaluator_mem0_thread_multiplier` and `evaluator_mem0_thread_cap` so evaluator workers stay close to memory capacity.
- Avoid `memory_operation_semaphore_limit: 1` for parallel evaluation; use `2` or higher.
- Set `memory_slot_timeout_ms` low (for example `250`-`500`) to avoid long blocking when memory slots are saturated.
- If timeout storms occur, lower `pressure_timeout_trip_threshold` and use `pressure_cooldown_ms` to let the backend recover.
- Keep `drop_store_on_backpressure: true` and `degrade_search_on_backpressure: true` for resilient long runs.
- If you need immediate stop, press `Ctrl+C` twice: first requests graceful cancel, second forces abort.
- For smoke tests, run a small subset first (`--subset 20`) to validate runtime behavior.

## Embedding Endpoint Returns Mixed `400` and `200`

Symptoms:

- Docker logs from embedding service show alternating `POST /v1/embeddings` `400` and `200`.
- Evaluation keeps running but memory recall quality degrades and fallback warnings increase.

Actions:

- Enable `memory_config.embedder_force_dimensionless_requests: true` for OpenAI-compatible embedding endpoints.
- Keep `memory_config.enable_dimension_fallback: true` as a secondary safety net.
- If running a long job, validate with a small subset first and confirm `400` spikes disappear before scaling up.

## MLflow `evaluate(...)` API Mismatch Errors

Symptoms:

- `unexpected keyword argument` errors for evaluation parameters.

Actions:

- Use the evaluation API signature matching the installed MLflow version.
- Prefer pinned MLflow versions in reproducible environments.
- Re-run with `uv sync` after dependency updates.

## `uv run evaluate.py` Uses All CPU Cores

Symptoms:

- Evaluation spikes to near-100% usage on high-core machines (for example, 64 cores).
- Host responsiveness degrades during MAS evaluation.

Actions:

- Set `evaluator_num_threads` in `configs/mas_config.yaml` to a bounded value (for example, `8`).
- Use deterministic backend for full control via `dspy.Parallel` threading.
- For GenAI backend, ensure `MLFLOW_GENAI_EVAL_MAX_WORKERS` is bounded; the evaluator now maps this from `evaluator_num_threads` by default.
- Reduce `--subset` during iterative debugging to minimize concurrent workload and latency.

## Sphinx Build Failures

Symptoms:

- Documentation build fails on warnings or unresolved references.

Actions:

- Run local docs build with warnings as errors:

```bash
uv run sphinx-build -M html docs docs/_build -W --keep-going
```

- Fix broken links/toctrees and verify Mermaid blocks are correctly fenced.

## Dataset Pipeline Inconsistencies

Symptoms:

- Unexpected split sizes or category imbalance.

Actions:

- Verify consistent `--seed` and `--train-ratio` values.
- Ensure scripts are run in order: download -> ingest -> unify -> split.
- Confirm the same DB URL is used across all script steps.

## Related Pages

- {doc}`/guides/how_to/scripts`
- {doc}`/guides/reference/reproducibility`
- {doc}`/guides/reference/metrics`
