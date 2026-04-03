# Data Pipeline and `src/data`

This page documents the `src/bias_mitigation/data` package and its role in the experiment workflow.

## Responsibilities

The data package provides:

- dataset schemas and models,
- configuration models for data and experiment loading,
- data ingestion and tracking helpers,
- stratified splitting utilities,
- repository interfaces for unified entries.

## Workflow Position

`src/bias_mitigation/data` powers the full preprocessing lifecycle:

1. load/ingest benchmark records,
2. normalize to unified MCQ-compatible entries,
3. stratify into train/dev splits,
4. expose reproducible examples for MAS evaluation.

## Key Submodules

- `bias_mitigation.data.models`: configuration and memory-related model schemas.
- `bias_mitigation.data.schemas`: SQLModel/Pydantic entities for dataset records.
- `bias_mitigation.data.dataset_tracker`: dataset tracking and MLflow logging helpers.
- `bias_mitigation.data.splitters`: stratified split logic for balanced evaluation.
- `bias_mitigation.data.repository`: unified dataset access layer.

## API Entry

For module-level API details, see:

- {doc}`/api/data`
