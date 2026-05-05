"""Reusable metric aggregation helpers for MAS evaluation pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from math import sqrt
from statistics import fmean, pstdev
from typing import Any


def mean_metric_dict(records: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean value per metric key across a list of metric dicts."""
    metric_keys = {key for record in records for key in record}
    return {
        key: fmean(record[key] for record in records if key in record)
        for key in metric_keys
    }


def ci95(values: list[float]) -> float:
    """Compute 95% confidence half-width for a list of values."""
    if len(values) <= 1:
        return 0.0
    return 1.96 * (pstdev(values) / sqrt(len(values)))


def metric_uncertainty(records: list[dict[str, float]]) -> dict[str, float]:
    """Compute per-metric 95% confidence half-width across metric records."""
    metric_keys = {key for record in records for key in record}
    return {
        key: ci95([record[key] for record in records if key in record])
        for key in metric_keys
    }


def aggregate_stratified_metric_rows(
    *,
    records: list[dict[str, Any]],
    stratify_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate records by stratification fields into support/metrics/ci95 rows.

    Each input record must contain:
    - `metadata`: dict[str, str]
    - `metrics`: dict[str, float]
    """

    def strata_key(record: dict[str, Any]) -> tuple[str, ...]:
        metadata = record['metadata']
        return tuple(str(metadata[field]) for field in stratify_fields)

    sorted_records = sorted(records, key=strata_key)
    rows: list[dict[str, Any]] = []
    for _key, grouped in groupby(sorted_records, key=strata_key):
        bucket = list(grouped)
        bucket_metrics = [dict(item['metrics']) for item in bucket]
        first_metadata = dict(bucket[0]['metadata'])
        rows.append(
            {
                'dimensions': {
                    field: str(first_metadata[field])
                    for field in stratify_fields
                },
                'support': len(bucket),
                'metrics': mean_metric_dict(bucket_metrics),
                'ci95': metric_uncertainty(bucket_metrics),
            }
        )
    return rows


def validate_stratified_dimensions(
    *,
    rows: list[dict[str, Any]],
    expected_fields: Sequence[str],
) -> None:
    """Ensure each aggregated row has exactly the configured dimensions."""
    expected = set(expected_fields)
    for row in rows:
        dimensions = row['dimensions']
        row_keys = set(dimensions.keys())
        if row_keys != expected:
            raise ValueError(
                'Stratified row has inconsistent dimensions: '
                f'expected={sorted(expected)}, got={sorted(row_keys)}'
            )
