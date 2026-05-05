"""Sequential pipeline wiring loader → validator → estimator → reporter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bias_mitigation.analysis.models import AnalysisConfig
from bias_mitigation.analysis.protocols import (
    EstimatorStrategy,
    LoaderStrategy,
    ReporterStrategy,
    ValidatorStrategy,
)


@dataclass(frozen=True)
class ScientificAnalysisPipeline:
    loader: LoaderStrategy
    validator: ValidatorStrategy
    estimator: EstimatorStrategy
    reporter: ReporterStrategy

    def execute(self, config: AnalysisConfig) -> tuple[Path, ...]:
        rows = self.loader.load(config)
        validated_rows = self.validator.validate(rows, config)
        estimates = self.estimator.estimate(validated_rows, config)
        return self.reporter.report(estimates, config)
