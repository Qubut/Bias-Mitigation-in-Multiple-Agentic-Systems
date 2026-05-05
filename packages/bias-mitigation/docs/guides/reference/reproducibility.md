# Reproducibility Checklist

Use this checklist to keep intervention comparisons valid and repeatable.

## Configuration Controls

- [ ] Fix intervention-independent settings (`num_agents`, `rounds`, `protocol`, model configs).
- [ ] Keep dataset sources and transformed schema constant across arms.
- [ ] Keep random seeds fixed for data sampling and split generation.

## Data Pipeline Controls

- [ ] Run scripts in canonical order (`download`, `ingest`, `unify`, `split`).
- [ ] Keep one DB URL consistently across pipeline stages.
- [ ] Version-control config files used for each run.

## Runtime Controls

- [ ] Use matched model endpoints and inference settings across interventions.
- [ ] Reset memory state between test cases in memory-based interventions.
- [ ] Log all runs to MLflow with intervention labels and config parameters.
- [ ] Enable local live stream persistence for long runs (`analysis_local_root`).
- [ ] Set durability policy (`stream_flush_every_events`, `stream_fsync`) for interruption safety.

## Live Analysis Artifacts

- Deterministic evaluation writes live per-event files under:
	- `analysis_local_root/<readable_run_dir>/stream_metric_rows.jsonl`
	- `analysis_local_root/<readable_run_dir>/stream_failure_rows.jsonl`
	- `analysis_local_root/<readable_run_dir>/stream_round_metrics.jsonl`
	- `analysis_local_root/<readable_run_dir>/stream_summary.json`
	- `analysis_local_root/<readable_run_dir>/run_manifest.json`
- Root index map for quick lookup:
	- `analysis_local_root/runs_index.jsonl`
- Naming and traceability are configurable via:
	- `analysis_live_dir_template`
	- `analysis_live_slug_max_length`
	- `analysis_live_write_manifest`
	- `analysis_live_index_filename`
- Final CSV/Parquet analysis tables remain logged to MLflow artifacts at `analysis_artifact_root`.

## Exclusions and Reporting

- [ ] Record technical exclusions with concrete reason codes.
- [ ] Report retained vs excluded counts.
- [ ] Provide sensitivity comparisons when exclusions occur.

## Validation Controls

- [ ] Run lint and tests before evaluation.
- [ ] Build docs with warnings as errors.
- [ ] Ensure diagram and reference pages are up to date for changed behavior.

## Related Pages

- {doc}`/guides/reference/metrics`
- {doc}`/guides/how_to/troubleshooting`
- {doc}`/guides/how_to/scripts`
