# {octicon}`database;1em` Data

`bias_mitigation.data` ingests **BBQ + StereoSet** into a unified
MCQ schema, splits into train / dev, and exposes the splits as
`dspy.Example` lists.

## {octicon}`workflow;1em` Pipeline

```mermaid
flowchart LR
    A[download-datasets] --> B[ingest-datasets]
    B --> C[unify-datasets]
    C --> D[split-datasets]
    D --> E[train.py / evaluate]
```

::::{tab-set}

:::{tab-item} download

Pulls BBQ + StereoSet source files into `datasets/`.
:::

:::{tab-item} ingest

Parses raw files into per-source SQL tables (`bias_mitigation.data.schemas`).
:::

:::{tab-item} unify

Transforms and samples entries into `UnifiedBiasEntry` records — a
common MCQ schema across both benchmarks.
:::

:::{tab-item} split

Stratified train / dev split, written as
`datasets/splits/{trainset,devset}.json`.
:::

::::

CLI invocations and per-script flags live in
[scripts](../how_to/scripts.md).

## {octicon}`package;1em` Submodules

| Module | Role |
|---|---|
| `bias_mitigation.data.loaders` | Async loaders for benchmark sources. |
| `bias_mitigation.data.schemas` | SQLModel tables for raw + unified rows. |
| `bias_mitigation.data.models` | Pydantic config (`MASConfig`, `Mem0Config`, dataset-config types). |
| `bias_mitigation.data.repository` | Single async DB session + CRUD helpers. |
| `bias_mitigation.data.splitters` | Stratified split + subset selection used by the evaluator. |
| `bias_mitigation.data.dataset_tracker` | MLflow `DatasetInput` builders + tag extraction. |

## {octicon}`sync;1em` Reproducibility

:::{important}
Fix `--seed` and `--train-ratio` across `split-datasets` invocations
that you intend to compare. Use the same `--db-url` across all four
scripts in one pipeline.
:::

## {octicon}`code-square;1em` API

[`api/data`](../../api/data.md).
