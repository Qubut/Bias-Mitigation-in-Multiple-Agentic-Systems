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

## MLflow `evaluate(...)` API Mismatch Errors

Symptoms:

- `unexpected keyword argument` errors for evaluation parameters.

Actions:

- Use the evaluation API signature matching the installed MLflow version.
- Prefer pinned MLflow versions in reproducible environments.
- Re-run with `uv sync` after dependency updates.

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
