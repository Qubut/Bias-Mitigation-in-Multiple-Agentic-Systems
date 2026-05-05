"""Concrete pipeline strategies: CSV loader, unit-interval validator, bootstrap estimator, JSON+CSV reporter."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

from bias_mitigation.analysis.models import (
    METRIC_NAMES,
    AnalysisConfig,
    DimensionValue,
    GroupEstimate,
    MetricEstimate,
    MetricRow,
)


def _safe_text(value: str | None) -> str:
    if value is None:
        return ''
    return value.strip()


def _safe_rate(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _build_row(raw_row: dict[str, str]) -> MetricRow | None:
    propagation_rate = _safe_rate(raw_row.get('MAS_Propagation_Rate'))
    system_robustness = _safe_rate(raw_row.get('MAS_System_Robustness'))
    emergence_rate = _safe_rate(raw_row.get('MAS_Emergence_Rate'))
    amplification_rate = _safe_rate(raw_row.get('MAS_Amplification_Rate'))
    if (
        propagation_rate is None
        or system_robustness is None
        or emergence_rate is None
        or amplification_rate is None
    ):
        return None
    return MetricRow(
        run_id=_safe_text(raw_row.get('run_id')),
        sample_id=_safe_text(raw_row.get('sample_id')),
        intervention=_safe_text(raw_row.get('intervention')),
        protocol=_safe_text(raw_row.get('protocol')),
        dataset_name=_safe_text(raw_row.get('dataset_name')),
        category=_safe_text(raw_row.get('category')),
        model_name=_safe_text(raw_row.get('model_name')),
        propagation_rate=propagation_rate,
        system_robustness=system_robustness,
        emergence_rate=emergence_rate,
        amplification_rate=amplification_rate,
    )


def _group_key(row: MetricRow, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(getattr(row, field) for field in fields)


def _bootstrap_mean(values: tuple[float, ...], rng: random.Random) -> float:
    sample_count = len(values)
    total = 0.0
    for _ in range(sample_count):
        index = rng.randrange(sample_count)
        total += values[index]
    return total / sample_count


def _bootstrap_interval(
    values: tuple[float, ...],
    bootstrap_samples: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    support = len(values)
    if support == 0:
        return (math.nan, math.nan, math.nan)

    mean_value = sum(values) / support
    if support == 1:
        return (mean_value, mean_value, mean_value)

    bootstrapped_means = [_bootstrap_mean(values, rng) for _ in range(bootstrap_samples)]

    bootstrapped_means.sort()
    low_index = int(0.025 * (bootstrap_samples - 1))
    high_index = int(0.975 * (bootstrap_samples - 1))
    return (
        mean_value,
        bootstrapped_means[low_index],
        bootstrapped_means[high_index],
    )


class CsvMetricRowLoader:
    def load(self, config: AnalysisConfig) -> tuple[MetricRow, ...]:
        rows: list[MetricRow] = []
        if not config.live_root.exists():
            return ()

        run_dirs = sorted(path for path in config.live_root.iterdir() if path.is_dir())
        for run_dir in run_dirs:
            metric_path = run_dir / 'stream_metric_rows.csv'
            if not metric_path.exists():
                continue
            with metric_path.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                for raw_row in reader:
                    typed_row = _build_row(raw_row)
                    if typed_row is None:
                        continue
                    rows.append(typed_row)
        return tuple(rows)


class UnitIntervalMetricValidator:
    """Silently drops rows where any metric falls outside [0, 1]."""

    def validate(self, rows: tuple[MetricRow, ...], config: AnalysisConfig) -> tuple[MetricRow, ...]:
        del config
        validated_rows: list[MetricRow] = []
        for row in rows:
            in_range = all(0.0 <= row.metric_value(metric_name) <= 1.0 for metric_name in METRIC_NAMES)
            if in_range:
                validated_rows.append(row)
        return tuple(validated_rows)


class BootstrapGroupEstimator:
    """Groups rows by config dimensions and computes mean + 95% CI via non-parametric bootstrap."""

    def estimate(self, rows: tuple[MetricRow, ...], config: AnalysisConfig) -> tuple[GroupEstimate, ...]:
        if not rows:
            return ()

        rng = random.Random(config.random_seed)
        buckets: dict[tuple[str, ...], list[MetricRow]] = defaultdict(list)
        for row in rows:
            buckets[_group_key(row, config.group_by)].append(row)

        group_estimates: list[GroupEstimate] = []
        for bucket_key, bucket_rows in sorted(buckets.items(), key=itemgetter(0)):
            dimensions = tuple(
                DimensionValue(field=field_name, value=field_value)
                for field_name, field_value in zip(config.group_by, bucket_key, strict=True)
            )
            estimates: list[MetricEstimate] = []
            for metric_name in METRIC_NAMES:
                values = tuple(row.metric_value(metric_name) for row in bucket_rows)
                mean_value, ci_low, ci_high = _bootstrap_interval(
                    values=values,
                    bootstrap_samples=config.bootstrap_samples,
                    rng=rng,
                )
                estimates.append(
                    MetricEstimate(
                        metric=metric_name,
                        mean=mean_value,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        support=len(values),
                    )
                )
            group_estimates.append(GroupEstimate(dimensions=dimensions, estimates=tuple(estimates)))
        return tuple(group_estimates)


JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list['JsonValue'] | dict[str, 'JsonValue']


def _estimate_to_json_payload(group_estimate: GroupEstimate) -> dict[str, JsonValue]:
    dimensions_payload: dict[str, JsonValue] = {
        dimension.field: dimension.value for dimension in group_estimate.dimensions
    }
    estimates_payload: list[JsonValue] = [
        {
            'metric': estimate.metric,
            'mean': estimate.mean,
            'ci_low': estimate.ci_low,
            'ci_high': estimate.ci_high,
            'support': estimate.support,
        }
        for estimate in group_estimate.estimates
    ]
    return {
        'dimensions': dimensions_payload,
        'estimates': estimates_payload,
    }


class JsonCsvReporter:
    def report(self, estimates: tuple[GroupEstimate, ...], config: AnalysisConfig) -> tuple[Path, ...]:
        config.output_root.mkdir(parents=True, exist_ok=True)

        json_path = config.output_root / 'group_estimates.json'
        csv_path = config.output_root / 'group_estimates.csv'

        json_payload = [_estimate_to_json_payload(estimate) for estimate in estimates]
        json_path.write_text(json.dumps(json_payload, indent=2), encoding='utf-8')

        with csv_path.open('w', encoding='utf-8', newline='') as handle:
            fieldnames = [*config.group_by, 'metric', 'mean', 'ci_low', 'ci_high', 'support']
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for group_estimate in estimates:
                dimension_map = {dimension.field: dimension.value for dimension in group_estimate.dimensions}
                for metric_estimate in group_estimate.estimates:
                    row_payload = {
                        **dimension_map,
                        'metric': metric_estimate.metric,
                        'mean': metric_estimate.mean,
                        'ci_low': metric_estimate.ci_low,
                        'ci_high': metric_estimate.ci_high,
                        'support': metric_estimate.support,
                    }
                    writer.writerow(row_payload)

        return (json_path, csv_path)
