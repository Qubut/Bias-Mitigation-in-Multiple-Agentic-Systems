# {octicon}`workflow;1em` Architecture

How one evaluation sample runs. Every claim points to a file or
function so it can be verified against the code.

## {octicon}`package;1em` Components

| Component | Module | Role |
|---|---|---|
| `MASConfig` | `data/models/config.py` | Pydantic root config; cross-field validators. |
| `Container` | `containers.py` | DI graph; per-intervention memory wiring; syncified `MASProgram` factory. |
| `MASProgram` | `mas/mas_program.py` | DSPy module; runs one debate; sync `forward` grafted via `dspy.syncify`. |
| `MASStateMachine` | `mas/mas_statemachine.py` | Per-sample lifecycle. |
| `Agent` + `AgentStateMachine` | `mas/agent.py`, `mas/agent_statemachine.py` | LM call + memory recall/store. Statechart owns predictor dispatch, tenacity retry, cross-thread LM bulkhead. |
| `Mem0Tools` | `memory/mem0_tools.py` | Async wrapper over `mem0.AsyncMemory`; recovery pipeline; purgatory breakers. |
| `MemoryOrchestrator` | `memory/orchestration/service.py` | Per-call timeouts; load-aware statechart. |
| `WorkflowMachine` | `workflows/statechart.py` | `prepare → build → execute → persist`. |
| `paper_bias_metrics_gepa` | `mas/metrics.py` | GEPA composite metric. |

## {octicon}`workflow;1em` State machines

The four diagrams below are regenerated from the class bodies by
`uv run generate-statecharts`. **Do not edit them
by hand** — re-run the script after editing any
`python-statemachine` subclass.

:::{dropdown} {octicon}`globe;1em` MAS lifecycle
:animate: fade-in-slide-down
:color: primary
:open:

```{include} ../../_generated/mas_statemachine.md
```

One `advance` event. `MASStateMachine.rounds_exhausted` is the
guard (`current_round > config.rounds`).
:::

:::{dropdown} {octicon}`person;1em` Agent lifecycle
:animate: fade-in-slide-down

```{include} ../../_generated/agent_statemachine.md
```

`allow_event_without_transition = False`
(`agent_statemachine.py`) so illegal sequences raise
`statemachine.exceptions.TransitionNotAllowed`.
:::

:::{dropdown} {octicon}`database;1em` Memory orchestration
:animate: fade-in-slide-down

```{include} ../../_generated/memory_orchestration_statechart.md
```

`note_success`, `note_failure`, `note_pressure` are the public
events. Counters reset on state entry (`on_enter_recovering`,
`on_enter_healthy`, `on_enter_shed`).
:::

:::{dropdown} {octicon}`gear;1em` Workflow
:animate: fade-in-slide-down

```{include} ../../_generated/workflow_machine.md
```

Each stage's runtime is a `WorkflowRuntime` (`workflows/statechart.py`).
:::

## {octicon}`broadcast;1em` Per-step LM dispatch

`AgentStateMachine.predictor_call` builds the zero-arg predictor
closure for the current phase. Two Pydantic envelopes
(`_InitialInputs`, `_UpdateInputs`) shape the kwargs.
`past_interaction_memory` is added only when the predictor
signature declares it (detected via `_predictor_input_fields`).

`AgentStateMachine.run_predictor` wraps that closure in:

- {octicon}`shield;1em` **Cross-thread bulkhead** —
  `threading.BoundedSemaphore` keyed on `(api_base, model)`.
  `asyncio.Semaphore` is loop-local and unsafe under
  `dspy.syncify`'s worker loops.
- {octicon}`sync;1em` **Retry** — `tenacity.AsyncRetrying` with
  exponential backoff; the blocking acquire is bridged via
  `asyncio.to_thread`.
- {octicon}`shield-check;1em` **Result envelope** —
  `try/except → returns.Result[dspy.Prediction, AgentExecutionError]`.
  Failures don't bubble; they're typed.

## {octicon}`database;1em` Memory stack

`Mem0Tools` awaits `mem0.AsyncMemory.search` / `add` directly. The
recovery pipeline:

- {octicon}`sync;1em` **Transient retries** — `tenacity.AsyncRetrying`.
- {octicon}`flame;1em` **Pressure / fallback breakers** —
  `purgatory.AsyncCircuitBreakerFactory`, materialised lazily via
  `asyncstdlib.functools.lru_cache`.
- {octicon}`shield;1em` **Cross-loop slot bulkhead** —
  `threading.BoundedSemaphore`, acquired through
  `asyncio.to_thread`.
- {octicon}`workflow;1em` **Result composition** —
  `returns.Result.bind/map/lash` composes the steps end-to-end.

`MemoryOrchestrator` adds `asyncio.wait_for` timeouts and the
`healthy → degraded → shed → recovering` statechart in
`memory/orchestration/statechart.py`.

## {octicon}`stack;1em` DI graph

`bias_mitigation.containers.Container`:

```python
mas_config          = providers.Dependency(instance_of=MASConfig)
memory_tools        = providers.Selector(mas_config.provided.intervention, ...)
memory_orchestrator = providers.Selector(mas_config.provided.intervention, ...)
mas_program         = providers.Factory(MASProgram.syncified, ...)
```

`_assert_intervention_coverage()` runs at import and raises
`RuntimeError` if any `InterventionType` member is missing from
either `Selector`.

`bias_mitigation.analysis.containers.AnalysisContainer.config` is a
`Singleton(load_analysis_config)` — defers `configs/analysis.yaml`
loading until first call.

## {octicon}`graph;1em` Analysis pipeline

`bias_mitigation.analysis.pipeline`:

```
load_runs            pl.concat([pl.read_csv(...)], how='diagonal_relaxed')
filter_unit_interval df.filter(pl.all_horizontal([...]))
group_estimates      df.group_by(...).agg(_bootstrap_ci × METRIC_COLS)
write_outputs        estimates.write_csv(...) + JSON envelope
```

`_bootstrap_ci` delegates to `library_stats.safe_bootstrap_mean_ci`
(scipy percentile bootstrap, ≥6 sample guard).

## {octicon}`file-binary;1em` Streaming sinks

`evaluation/analysis/live/<run_dir>/`:

| File | Contents |
|---|---|
| `stream_metric_rows.csv` | Per-sample metrics. |
| `stream_failure_rows.jsonl` | Failures with full traceback (formatted by `mas/evaluator.py::metric`). |
| `stream_round_metrics.csv` | Per-round bias attribution. |

## {octicon}`link;1em` See also

- [Maintenance contract](../developer/maintenance.md) — code edits
  that require doc updates.
