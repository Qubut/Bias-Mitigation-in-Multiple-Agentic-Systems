# Scientific Analysis Pipeline

This package provides a declarative and strongly typed analysis pipeline for MAS metric rows.

## Architecture

- `CsvMetricRowLoader`: loads `stream_metric_rows.csv` across live run directories.
- `UnitIntervalMetricValidator`: enforces bounded metric semantics in `[0, 1]`.
- `BootstrapGroupEstimator`: computes grouped means and bootstrap confidence intervals.
- `JsonCsvReporter`: exports typed outputs to JSON and CSV.
- `ScientificAnalysisPipeline`: composes strategies declaratively.

## Run

```bash
uv run analyze-scientific
```

Custom options:

```bash
uv run analyze-scientific \
  --live-root evaluation/analysis/live \
  --output-root evaluation/analysis/scientific_notebook_outputs \
  --group-by intervention,protocol,dataset_name,model_name \
  --bootstrap-samples 3000 \
  --random-seed 42
```

## Outputs

- `group_estimates.json`
- `group_estimates.csv`
