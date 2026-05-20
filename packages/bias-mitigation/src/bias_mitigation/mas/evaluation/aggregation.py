"""Group-level metric aggregation for the MAS fairness report.

The deterministic evaluator produces one record per sample carrying
already-computed scalar metrics (system robustness, emergence,
amplification, propagation) and an optional ``(y_true, y_pred)`` pair
representing the gold label and the agents' majority-vote answer.  This
module rolls those records up by the evaluator's stratification fields
(dataset, demographic category, intervention, seed, …) so the run's
output table has one row per stratum.

Two aggregation paths are interleaved on the same input:

* **MAS scalar metrics** are averaged per group with polars and reported
  alongside a normal-approximation 95% confidence half-width.  Polars is
  used (rather than ``MetricFrame``) because these metrics are already
  scalars; ``MetricFrame`` is designed to *compute* metrics from
  predictions, not to re-aggregate scalars.
* **Classification fairness** is computed with
  :class:`fairlearn.metrics.MetricFrame`: per-group accuracy plus the
  run-wide ``demographic_parity_difference`` and
  ``equalized_odds_difference`` disparities.  These attach to every
  stratified row so downstream consumers can answer "is the MAS equally
  accurate across groups?" without re-deriving anything.

Records missing ``y_true`` or ``y_pred`` are silently dropped from the
fairness path but still contribute to the MAS-metric aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np
import polars as pl
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score


def mean_metric_dict(records: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric key across a list of per-sample metric dicts.

    Used by the evaluator's overall-aggregate path (across all strata) so
    the run summary carries a single mean per metric.  Keys that appear in
    some records but not others are averaged over the records that do
    contain them — there is no implicit zero-fill.

    Args:
        records: One metric dict per sample.  Empty list yields ``{}``.

    Returns:
        Metric name → arithmetic mean over present values, sorted by key.
    """
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    return {key: float(np.mean([r[key] for r in records if key in r])) for key in keys}


def metric_uncertainty(records: list[dict[str, float]]) -> dict[str, float]:
    """Per-metric 95% confidence half-width of the sample mean.

    Computed as ``1.96 * pstdev / sqrt(n)`` — the normal-approximation
    interval reviewers expect to see beside a reported mean.  Single-
    sample groups yield ``0.0`` because the half-width is undefined and
    reporting ``NaN`` breaks downstream JSON/CSV consumers.

    Args:
        records: One metric dict per sample.  Empty list yields ``{}``.

    Returns:
        Metric name → half-width of the 95% CI on its sample mean, sorted
        by key.
    """
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    return {
        key: (
            0.0
            if len(vals := np.asarray([r[key] for r in records if key in r], dtype=float)) <= 1
            else float(1.96 * vals.std(ddof=0) / np.sqrt(len(vals)))
        )
        for key in keys
    }


_FAIRNESS_DISPARITY_KEYS = (
    'demographic_parity_difference',
    'equalized_odds_difference',
)


def aggregate_stratified_metric_rows(
    *,
    records: list[dict[str, Any]],
    stratify_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Roll per-sample records up into one row per stratification group.

    Drives the evaluator's stratified report: every group identified by
    ``stratify_fields`` becomes a row carrying the support count, the
    mean of each MAS metric, its 95% CI half-width, and — when label
    information is available — Fairlearn's accuracy and disparity
    statistics.

    Args:
        records: Per-sample dicts shaped as
            ``{'metadata': {...}, 'metrics': {...}}``.  ``metadata`` must
            contain every key in ``stratify_fields``; missing values
            stratify as the string ``"unknown"`` so a malformed sample
            does not silently disappear.  ``metadata['y_true']`` /
            ``metadata['y_pred']`` are optional and only used by the
            fairness path.
        stratify_fields: Ordered tuple of metadata keys defining each
            stratum.  The output row's ``dimensions`` dict carries
            exactly these keys.

    Returns:
        One ``dict`` per stratum with keys ``dimensions``, ``support``,
        ``metrics``, ``ci95`` — the shape consumed by
        :class:`_StratifiedRow`.  Empty input yields ``[]``.

    Notes:
        - Per-group ``fairlearn_accuracy`` is only added for groups whose
          records all carry ``y_true``/``y_pred``.
        - The run-wide ``demographic_parity_difference`` /
          ``equalized_odds_difference`` scalars are broadcast onto every
          row so a single stratum can be filtered without losing the
          fairness context.
    """
    if not records:
        return []

    stratify_fields = list(stratify_fields)
    metric_keys = sorted({key for record in records for key in record['metrics']})
    frame = pl.DataFrame([
        {
            **{field: str(record['metadata'].get(field, 'unknown')) for field in stratify_fields},
            **{key: float(record['metrics'].get(key, 0.0)) for key in metric_keys},
            'y_true': record['metadata'].get('y_true'),
            'y_pred': record['metadata'].get('y_pred'),
        }
        for record in records
    ])

    mean_suffix, std_suffix = '__mean', '__std'
    grouped = frame.group_by(stratify_fields, maintain_order=True).agg(
        pl.len().alias('support'),
        *[pl.col(k).mean().alias(f'{k}{mean_suffix}') for k in metric_keys],
        *[pl.col(k).std(ddof=0).fill_null(0.0).alias(f'{k}{std_suffix}') for k in metric_keys],
    )

    accuracy = _per_group_accuracy(frame, stratify_fields)
    disparity = _run_level_fairness_disparity(frame, stratify_fields)

    def _build_row(row: dict[str, Any]) -> dict[str, Any]:
        dims_tuple = tuple(str(row[field]) for field in stratify_fields)
        support = int(row['support'])
        group_metrics: dict[str, float] = {
            key: float(row[f'{key}{mean_suffix}']) for key in metric_keys
        }
        group_ci: dict[str, float] = {
            key: float(1.96 * row[f'{key}{std_suffix}'] / sqrt(support)) if support > 1 else 0.0
            for key in metric_keys
        }
        if dims_tuple in accuracy:
            group_metrics['fairlearn_accuracy'] = accuracy[dims_tuple]
            group_ci['fairlearn_accuracy'] = 0.0
        for key, value in disparity.items():
            group_metrics[key] = value
            group_ci[key] = 0.0
        return {
            'dimensions': {field: str(row[field]) for field in stratify_fields},
            'support': support,
            'metrics': group_metrics,
            'ci95': group_ci,
        }

    return [_build_row(row) for row in grouped.iter_rows(named=True)]


def _per_group_accuracy(
    frame: pl.DataFrame,
    stratify_fields: list[str],
) -> dict[tuple[str, ...], float]:
    """Compute classification accuracy for each stratification group.

    Args:
        frame: Long-format polars DataFrame with one row per sample.
            Must carry the ``y_true``/``y_pred`` columns and every name
            in ``stratify_fields``.
        stratify_fields: Columns of ``frame`` that define each group; the
            same list used for the run's stratified output.

    Returns:
        Map from a stratum's value tuple (in ``stratify_fields`` order) to
        its accuracy.  Empty when no row carries both ``y_true`` and
        ``y_pred`` — the caller treats that as "no fairness columns" and
        emits MAS-only metrics.
    """
    labelled = frame.drop_nulls(subset=['y_true', 'y_pred'])
    if labelled.is_empty():
        return {}
    sensitive = labelled.select(stratify_fields).cast(pl.Utf8).to_pandas()
    metric_frame = MetricFrame(
        metrics={'accuracy': accuracy_score},
        y_true=labelled['y_true'].cast(pl.Int64).to_numpy(),
        y_pred=labelled['y_pred'].cast(pl.Int64).to_numpy(),
        sensitive_features=sensitive,
    )
    by_group = metric_frame.by_group['accuracy']
    return {
        (key if isinstance(key, tuple) else (str(key),)): float(value)
        for key, value in by_group.items()
    }


def _run_level_fairness_disparity(
    frame: pl.DataFrame,
    stratify_fields: list[str],
) -> dict[str, float]:
    """Compute run-wide disparity statistics across all sensitive groups.

    Reports the worst-case-vs-best-case spread in selection rate
    (``demographic_parity_difference``) and the spread of TPR/FPR
    (``equalized_odds_difference``).  Both are scalar summaries of the
    entire run — they answer "how unfair is the MAS overall?" — and are
    broadcast onto every stratified row so a single stratum filter still
    surfaces them.

    Args:
        frame: Long-format polars DataFrame with one row per sample.
        stratify_fields: Columns of ``frame`` used as Fairlearn's
            ``sensitive_features``.

    Returns:
        ``{'demographic_parity_difference': ...,
        'equalized_odds_difference': ...}`` as scalars.  Empty when no
        row carries both ``y_true`` and ``y_pred`` *or* every labelled
        row falls into a single sensitive group (disparity needs at
        least two groups to be defined).
    """
    labelled = frame.drop_nulls(subset=['y_true', 'y_pred'])
    if labelled.is_empty():
        return {}
    sensitive = labelled.select(stratify_fields).cast(pl.Utf8)
    if sensitive.unique().height < 2:
        return {}
    sensitive_pd = sensitive.to_pandas()
    y_true = labelled['y_true'].cast(pl.Int64).to_numpy()
    y_pred = labelled['y_pred'].cast(pl.Int64).to_numpy()
    return {
        'demographic_parity_difference': float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_pd)
        ),
        'equalized_odds_difference': float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_pd)
        ),
    }


def validate_stratified_dimensions(
    *,
    rows: list[dict[str, Any]],
    expected_fields: Sequence[str],
) -> None:
    """Assert every aggregated row carries exactly the configured dimensions.

    Acts as a tripwire on the contract between the evaluator's
    stratification config and the rows it emits: if a record's metadata
    is silently missing a key (or carries an unexpected one) downstream
    fairness tables would compare apples to oranges.

    Args:
        rows: Aggregated rows as returned by
            :func:`aggregate_stratified_metric_rows`.
        expected_fields: The exact set of dimension keys every row must
            carry.

    Raises:
        ValueError: As soon as a row's dimension keys differ from
            ``expected_fields``.
    """
    expected = set(expected_fields)
    for row in rows:
        dimensions = row['dimensions']
        row_keys = set(dimensions.keys())
        if row_keys != expected:
            raise ValueError(
                'Stratified row has inconsistent dimensions: '
                f'expected={sorted(expected)}, got={sorted(row_keys)}'
            )
