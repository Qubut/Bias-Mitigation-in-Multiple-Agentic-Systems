"""Live-runs analysis pipeline as a single polars chain.

``load_runs → filter_unit_interval → group_estimates → write_outputs``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from bias_mitigation.analysis.library_stats import safe_bootstrap_mean_ci

#: Columns the streaming sink writes as per-sample metrics — single source of
#: truth for both the unit-interval filter and the group estimator.
METRIC_COLS: Final[tuple[str, ...]] = (
    'MAS_Propagation_Rate',
    'MAS_System_Robustness',
    'MAS_Emergence_Rate',
    'MAS_Amplification_Rate',
)

#: Default stratification — every column must exist in the streamed CSV.
DEFAULT_GROUP_BY: Final[tuple[str, ...]] = (
    'intervention',
    'protocol',
    'dataset_name',
    'model_name',
)


@dataclass(frozen=True)
class LiveAnalysisConfig:
    """Inputs to :func:`run_analysis`."""

    live_root: Path
    output_root: Path
    group_by: tuple[str, ...] = DEFAULT_GROUP_BY
    bootstrap_samples: int = 2000
    random_seed: int = 42


def load_runs(live_root: Path) -> pl.DataFrame:
    """Concatenate every ``<run>/stream_metric_rows.csv`` under ``live_root``.

    Uses ``how='diagonal_relaxed'`` so missing columns from interrupted
    runs become nulls instead of raising — the unit-interval filter then
    drops them.  Returns an empty frame when no run directories exist.
    """
    csv_paths = sorted(live_root.glob('*/stream_metric_rows.csv'))
    if not csv_paths:
        return pl.DataFrame()
    return pl.concat(
        [pl.read_csv(p, infer_schema_length=10_000, ignore_errors=True) for p in csv_paths],
        how='diagonal_relaxed',
    )


def filter_unit_interval(df: pl.DataFrame, metric_cols: tuple[str, ...] = METRIC_COLS) -> pl.DataFrame:
    """Keep rows where every named metric is non-null and in ``[0, 1]``."""
    if df.is_empty():
        return df
    return df.filter(
        pl.all_horizontal([
            pl.col(name).is_not_null() & pl.col(name).is_between(0, 1)
            for name in metric_cols
        ]),
    )


def _bootstrap_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)``; degrades to ``mean=lo=hi`` when ``n<6``."""
    if values.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    if values.size < 6:
        m = float(values.mean())
        return (m, m, m)
    ci = safe_bootstrap_mean_ci(values, n_resamples=n_resamples, seed=seed)
    return (ci.mean, ci.lo, ci.hi)


def group_estimates(df: pl.DataFrame, config: LiveAnalysisConfig) -> pl.DataFrame:
    """Long-format ``(*group_by, metric, mean, ci_low, ci_high, support)`` table.

    One row per ``(group, metric)`` pair so downstream tools (notebooks,
    fairness reports, CSV consumers) read a tidy long frame instead of
    nested per-group records.
    """
    if df.is_empty():
        return pl.DataFrame()

    return pl.DataFrame([
        {
            **dict(zip(config.group_by, group_key, strict=True)),
            'metric': metric,
            **dict(zip(['mean', 'ci_low', 'ci_high'], _bootstrap_ci(
                group_df[metric].drop_nulls().to_numpy(),
                config.bootstrap_samples,
                config.random_seed,
            ), strict=True)),
            'support': group_df.height,
        }
        for group_key, group_df in df.group_by(list(config.group_by), maintain_order=True)
        for metric in METRIC_COLS
    ])


def write_outputs(estimates: pl.DataFrame, config: LiveAnalysisConfig) -> tuple[Path, ...]:
    """Write ``group_estimates.{csv,json}`` under ``config.output_root``.

    CSV is the tidy long frame as-is.  JSON nests the group-by columns
    under a ``dimensions`` key to match the previous reporter's shape so
    notebook consumers don't break.
    """
    config.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_root / 'group_estimates.csv'
    json_path = config.output_root / 'group_estimates.json'
    estimates.write_csv(csv_path)
    json_payload = [
        {
            'dimensions': {dim: record[dim] for dim in config.group_by},
            'estimates': [
                {
                    'metric': record['metric'],
                    'mean': record['mean'],
                    'ci_low': record['ci_low'],
                    'ci_high': record['ci_high'],
                    'support': record['support'],
                },
            ],
        }
        for record in estimates.to_dicts()
    ]
    json_path.write_text(json.dumps(json_payload, indent=2), encoding='utf-8')
    return (json_path, csv_path)


def run_analysis(config: LiveAnalysisConfig) -> tuple[Path, ...]:
    """End-to-end: load → filter → group-estimate → write."""
    df = load_runs(config.live_root)
    df = filter_unit_interval(df)
    estimates = group_estimates(df, config)
    return write_outputs(estimates, config)
