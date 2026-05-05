"""Strategy protocols for each stage of the analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from bias_mitigation.analysis.models import AnalysisConfig, GroupEstimate, MetricRow


class LoaderStrategy(Protocol):
    def load(self, config: AnalysisConfig) -> tuple[MetricRow, ...]:
        ...


class ValidatorStrategy(Protocol):
    def validate(self, rows: tuple[MetricRow, ...], config: AnalysisConfig) -> tuple[MetricRow, ...]:
        ...


class EstimatorStrategy(Protocol):
    def estimate(self, rows: tuple[MetricRow, ...], config: AnalysisConfig) -> tuple[GroupEstimate, ...]:
        ...


class ReporterStrategy(Protocol):
    def report(self, estimates: tuple[GroupEstimate, ...], config: AnalysisConfig) -> tuple[Path, ...]:
        ...
