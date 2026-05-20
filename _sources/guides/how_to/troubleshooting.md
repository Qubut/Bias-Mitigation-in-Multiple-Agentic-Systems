# {octicon}`bug;1em` Troubleshooting

Each section below is a self-contained recipe — symptom → root
cause → fix.

:::{dropdown} {octicon}`alert;1em` `mem0g` run hangs at startup
:animate: fade-in-slide-down
:color: warning

Likely **mem0 telemetry blocking on SIGINT**. Confirm
`MEM0_TELEMETRY=False`:

```bash
MEM0_TELEMETRY=False uv run train --intervention mem0g
```

Then validate vector-store reachability (chroma path is writable,
or the qdrant/postgres URL is reachable) and the embedder endpoint
responds to `POST /v1/embeddings`.
:::

:::{dropdown} {octicon}`cpu;1em` `mem0g` saturates CPU cores
:animate: fade-in-slide-down
:color: warning

The evaluator runs through `dspy.Evaluate` with N worker threads;
`mem0g` adds embedder + vector-store I/O on top. Cap:

```yaml
evaluator_concurrency:
  max_evaluation_threads: 8
  max_llm_inflight_per_endpoint: 8
```

For `mem0g` / `mem0g_gepa` interventions, `evaluation.py`
additionally sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` **defensively** (only
when the env var is unset).
:::

:::{dropdown} {octicon}`x-circle;1em` `'tuple' object has no attribute 'get'` in failure rows
:animate: fade-in-slide-down
:color: danger

Every sample failed with the same `AttributeError`. The traceback
is preserved in `stream_failure_rows.jsonl::error` since
`mas/evaluator.py::metric` formats the traceback into that field.

:::{tip}
Read the **JSONL** row rather than the CSV — CSV collapses
newlines and truncates the traceback.
:::
:::

:::{dropdown} {octicon}`clock;1em` Recall / store timeouts dominate the log
:animate: fade-in-slide-down
:color: warning

`Mem0Tools._memory_slot` uses a cross-thread
`threading.BoundedSemaphore` — `asyncio.Semaphore` is loop-local
and breaks under `dspy.syncify`'s per-call event loops.

If you still see
`[MemoryOrchestrator]: recall timed out after Xs`, the **embedder**
is the likely bottleneck. Raise:

```yaml
memory_orchestration:
  recall_timeout_ms: 8000
  store_timeout_ms: 6000

memory_config:
  memory_slot_timeout_ms: 4000   # must be < recall_timeout_ms
```
:::

:::{dropdown} {octicon}`flame;1em` Pressure circuit opens immediately
:animate: fade-in-slide-down
:color: danger

The pressure breaker trips on consecutive
`_MemoryBackpressureError` events. Inspect
`Mem0Tools.stats_snapshot()` counters:

- `semaphore.wait_timeouts` high → raise
  `memory_operation_semaphore_limit` and `memory_slot_timeout_ms`.
- `search.fallback_circuit_open_skips` high → the **search-fallback**
  circuit (separate from pressure) tripped; extend
  `search_fallback_cooldown_ms` or raise
  `search_fallback_consecutive_fail_trip_threshold`.
:::

:::{dropdown} {octicon}`pulse;1em` Embedding endpoint flips between 400 and 200
:animate: fade-in-slide-down
:color: warning

The OpenAI-compatible embedder rejected the dimension override:

```yaml
memory_config:
  embedder_force_dimensionless_requests: true
  enable_dimension_fallback: true
```

`Mem0Tools._attempt_dimension_fallback_reinit` triggers a one-shot
re-init without the explicit dimension once a dim-mismatch error
is seen.
:::

:::{dropdown} {octicon}`book;1em` Sphinx build fails
:animate: fade-in-slide-down
:color: warning

```bash
LC_ALL=C.UTF-8 uv run sphinx-build -M html docs docs/_build -W --keep-going
```

Then fix any unresolved references or broken toctrees. Regenerate
state-machine diagrams when the build complains about missing
`docs/_generated/*.md`:

```bash
uv run generate-statecharts
```
:::

:::{dropdown} {octicon}`workflow;1em` Dataset script ordering
:animate: fade-in-slide-down
:color: primary

Run in order:
`download-datasets → ingest-datasets → unify-datasets →
split-datasets`. Use the **same** `--db-url` across all four,
the **same** `--seed` and `--train-ratio` across `split-datasets`
invocations within one comparison.

See [Scripts](scripts.md).
:::
