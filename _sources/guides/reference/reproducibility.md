# {octicon}`sync;1em` Reproducibility

A checklist for keeping intervention comparisons valid across runs.

::::{tab-set}

:::{tab-item} {octicon}`gear;1em` Configuration

- [ ] Fix `num_agents`, `rounds`, `protocol`, `agent_models` across
      arms in one comparison.
- [ ] Keep `gepa.seed` (default `42`) fixed.
- [ ] Keep dataset splits fixed —
      `split-datasets --seed --train-ratio` identical across
      pipeline rebuilds.
- [ ] Pin `MASConfig.intervention` via CLI
      (`--intervention <arm>`), not via YAML edits, when iterating.
:::

:::{tab-item} {octicon}`database;1em` Data pipeline

- [ ] Run scripts in order:
      `download-datasets → ingest-datasets → unify-datasets →
      split-datasets`.
- [ ] Use the same `--db-url` across all four.
- [ ] Version the YAML configs that produced the runs.
:::

:::{tab-item} {octicon}`broadcast;1em` Runtime

- [ ] Match model endpoints + temperatures across arms.
- [ ] For memory arms, memory is reset between samples at runtime
      — do **not** override `reset_memory_on_genesis: false`.
- [ ] Log every run to MLflow with the intervention tag set.
:::

:::{tab-item} {octicon}`shield-check;1em` Validation

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/bias_mitigation`
- [ ] `LC_ALL=C.UTF-8 uv run sphinx-build -M html docs docs/_build`
- [ ] `uv run generate-statecharts`
:::

::::

## {octicon}`file-binary;1em` Live streaming artefacts

Each evaluator run writes to `evaluation/analysis/live/<run_dir>/`:

| File | Contents |
|---|---|
| `stream_metric_rows.{csv,jsonl}` | Per-sample MAS scorer values. |
| `stream_failure_rows.{csv,jsonl}` | Per-sample failures with full traceback in `error`. |
| `stream_round_metrics.{csv,jsonl}` | Per-round bias attribution. |
| `stream_summary.json` + `run_manifest.json` | Counts + final summary; run id, intervention, agent map. |

A top-level `evaluation/analysis/live/runs_index.jsonl` indexes
every `run_dir`. CSV and JSONL are written in parallel; the polars
analysis pipeline reads the CSVs.

:::{dropdown} {octicon}`gear;1em` Streaming knobs in `mas_config.yaml`
:animate: fade-in-slide-down

- `analysis_local_root` — base path (default
  `evaluation/analysis/live`).
- `analysis_live_dir_template` — slug template for `<run_dir>`.
- `analysis_live_index_filename` — index filename.
- `stream_flush_every_events`, `stream_fsync` — durability vs
  throughput.
:::

## {octicon}`x-circle;1em` Exclusions

- [ ] Record technical exclusions with reason codes captured from
      `stream_failure_rows.jsonl::error`.
- [ ] Report retained vs excluded counts in any aggregate.

## {octicon}`link;1em` See also

- [Metrics](metrics.md)
- [Troubleshooting](../how_to/troubleshooting.md)
- [Scripts](../how_to/scripts.md)
