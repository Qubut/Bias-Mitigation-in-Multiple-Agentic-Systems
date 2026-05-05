"""CLI entrypoint for running the scientific analysis pipeline over live evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from bias_mitigation.analysis import (
    AnalysisConfig,
    BootstrapGroupEstimator,
    CsvMetricRowLoader,
    JsonCsvReporter,
    ScientificAnalysisPipeline,
    UnitIntervalMetricValidator,
)


def _parse_group_by(raw_value: str) -> tuple[str, ...]:
    fields = tuple(part.strip() for part in raw_value.split(',') if part.strip())
    if not fields:
        raise click.BadParameter('group-by must contain at least one field.')
    return fields


@click.command()
@click.option('--live-root', default='evaluation/analysis/live', show_default=True, type=click.Path(path_type=Path))
@click.option(
    '--output-root',
    default='evaluation/analysis/scientific_notebook_outputs',
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option('--group-by', default='intervention,protocol,dataset_name,model_name', show_default=True)
@click.option('--bootstrap-samples', default=2000, show_default=True, type=int)
@click.option('--random-seed', default=42, show_default=True, type=int)
def main(
    live_root: Path,
    output_root: Path,
    group_by: str,
    bootstrap_samples: int,
    random_seed: int,
) -> None:
    config = AnalysisConfig(
        live_root=live_root,
        output_root=output_root,
        group_by=_parse_group_by(group_by),
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )

    pipeline = ScientificAnalysisPipeline(
        loader=CsvMetricRowLoader(),
        validator=UnitIntervalMetricValidator(),
        estimator=BootstrapGroupEstimator(),
        reporter=JsonCsvReporter(),
    )

    outputs = pipeline.execute(config)
    logger.success('Scientific analysis artifacts generated.')
    for artifact_path in outputs:
        logger.info(f'Wrote: {artifact_path}')


if __name__ == '__main__':
    main()
