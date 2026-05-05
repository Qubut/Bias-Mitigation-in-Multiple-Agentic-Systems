"""Immutable data models for analysis inputs, per-sample metric rows, and bootstrap estimates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

MetricName = Literal[
    'MAS_Propagation_Rate',
    'MAS_System_Robustness',
    'MAS_Emergence_Rate',
    'MAS_Amplification_Rate',
]

METRIC_NAMES: Final[tuple[MetricName, ...]] = (
    'MAS_Propagation_Rate',
    'MAS_System_Robustness',
    'MAS_Emergence_Rate',
    'MAS_Amplification_Rate',
)

DEFAULT_GROUP_BY: Final[tuple[str, ...]] = (
    'intervention',
    'protocol',
    'dataset_name',
    'model_name',
)


@dataclass(frozen=True)
class AnalysisConfig:
    live_root: Path
    output_root: Path
    group_by: tuple[str, ...] = DEFAULT_GROUP_BY
    bootstrap_samples: int = 2000
    random_seed: int = 42


@dataclass(frozen=True)
class MetricRow:
    """One sample's per-run metrics as loaded from a CSV row."""

    run_id: str
    sample_id: str
    intervention: str
    protocol: str
    dataset_name: str
    category: str
    model_name: str
    propagation_rate: float
    system_robustness: float
    emergence_rate: float
    amplification_rate: float

    def metric_value(self, metric_name: MetricName) -> float:
        if metric_name == 'MAS_Propagation_Rate':
            return self.propagation_rate
        if metric_name == 'MAS_System_Robustness':
            return self.system_robustness
        if metric_name == 'MAS_Emergence_Rate':
            return self.emergence_rate
        return self.amplification_rate


@dataclass(frozen=True)
class DimensionValue:
    field: str
    value: str


@dataclass(frozen=True)
class MetricEstimate:
    """Bootstrap mean + 95% CI for a single metric over a group."""

    metric: MetricName
    mean: float
    ci_low: float
    ci_high: float
    support: int


@dataclass(frozen=True)
class GroupEstimate:
    dimensions: tuple[DimensionValue, ...]
    estimates: tuple[MetricEstimate, ...]
