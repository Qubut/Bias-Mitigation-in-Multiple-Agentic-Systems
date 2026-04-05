"""Packaged CLI entrypoint for evaluation runs."""

from __future__ import annotations

import importlib.util
import logging
import os
import signal
import time
import warnings
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import click
import dspy
import mlflow
from loguru import logger
from mlflow.models import EvaluationResult, MetricThreshold
from requests.exceptions import RequestsDependencyWarning

from bias_mitigation.containers import Container
from bias_mitigation.data.dataset_tracker import load_and_track_splits
from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.evaluator import EvaluatorBackend, MASEvaluator

os.environ['MEM0_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLED'] = 'true'
os.environ['DO_NOT_TRACK'] = '1'
if '/usr/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f'{os.environ.get("PATH", "")}:/usr/bin:/bin'

warnings.filterwarnings(
    'ignore', message='There is no current event loop', category=DeprecationWarning
)
warnings.filterwarnings('ignore', category=RequestsDependencyWarning)
logging.getLogger('mlflow.utils.git_utils').setLevel(logging.ERROR)


def setup_mlflow(tracking_uri: str, experiment_prefix: str) -> None:
    """Configure MLflow tracking and create a timestamped experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.active_run():
        mlflow.end_run()
    exp_name = f'{experiment_prefix}_{datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")}'
    mlflow.set_experiment(exp_name)


def _force_exit_handler(sig, frame) -> None:
    """Forcefully end active MLflow run on SIGINT to avoid stuck teardown."""
    logger.warning('⚠️  KeyboardInterrupt (Ctrl+C) trapped via SIGINT. Ending run...')
    if mlflow.active_run():
        mlflow.end_run(status='KILLED')
    os._exit(130)


@contextmanager
def single_parent_evaluation_run(run_name: str):
    """Guarantee one parent MLflow run for the full evaluation sequence."""
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f'🚀 Started parent run: {run.info.run_id}')
        yield run
        logger.info(f'✅ Parent run completed: {run.info.run_id}')


def run_optional_safety_scan(enabled: bool) -> dict[str, str]:
    """Run optional safety scan hook and return a status payload."""
    if not enabled:
        return {'enabled': 'false', 'status': 'skipped'}

    if importlib.util.find_spec('giskard') is None:
        logger.warning('⚠️ Safety scan requested but `giskard` is not installed.')
        return {'enabled': 'true', 'status': 'missing_dependency', 'dependency': 'giskard'}

    logger.info('✅ Safety scan hook available (`giskard` import succeeded).')
    return {'enabled': 'true', 'status': 'available'}


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option(
    '--config-path',
    type=click.Path(exists=True, path_type=Path),
    default=Path('configs/mas_config.yaml'),
)
@click.option(
    '--dataset-dir',
    type=click.Path(path_type=Path),
    default=Path('datasets/splits'),
)
@click.option('--tracking-uri', type=str, default='http://127.0.0.1:5003')
@click.option('--run-name', type=str, default='MAS_Experiment_Cooperative')
@click.option(
    '--intervention',
    type=click.Choice(['baseline', 'baseline_prompt_opt', 'mem0g', 'mem0g_gepa']),
    default=None,
    help='Override intervention strategy from config.',
)
@click.option(
    '--memory-config',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Optional memory config YAML merged after base config.',
)
@click.option(
    '--evaluator-backend',
    type=click.Choice(['deterministic', 'genai']),
    default='deterministic',
    show_default=True,
    help='Evaluation backend: deterministic metrics or MLflow GenAI scorers.',
)
@click.option(
    '--min-system-robustness',
    type=float,
    default=None,
    help='Optional minimum threshold for MAS_System_Robustness quality gate.',
)
@click.option(
    '--run-safety-scan',
    is_flag=True,
    default=False,
    help='Enable optional safety-layer integration hook (Giskard availability check).',
)
@click.option(
    '--subset',
    type=int,
    default=2250,
    show_default=True,
    help='Number of dev examples to evaluate (use small value for runtime validation).',
)
def main(
    config_path: Path,
    dataset_dir: Path,
    tracking_uri: str,
    run_name: str,
    intervention: str | None,
    memory_config: Path | None,
    evaluator_backend: str,
    min_system_robustness: float | None,
    run_safety_scan: bool,
    subset: int,
) -> None:
    """Run evaluation for the configured intervention with MLflow observability."""
    logger.info('Starting pipeline execution via Click CLI...')

    signal.signal(signal.SIGINT, _force_exit_handler)

    overrides = {'intervention': intervention} if intervention else {}
    paths_to_load = [str(config_path)]
    if memory_config:
        paths_to_load.append(str(memory_config))

    logger.debug('Loading configuration...')
    mas_config = MASConfig.load_merged(*paths_to_load, cli_overrides=overrides).unwrap()
    logger.debug('Configuration loaded successfully.')

    configured_memory = mas_config.memory_config
    memory_config_payload = (
        configured_memory.model_dump(mode='json', exclude_none=True)
        if configured_memory is not None
        else {}
    )

    container = Container()
    # Official dependency-injector pattern: load validated Pydantic config
    container.config.from_dict({
        'mas_config': mas_config.model_dump(mode='json', exclude_none=True),
        'memory_config': memory_config_payload,
        'protocol': mas_config.protocol,
    })
    # Wires all packages so MASProgram, Agent, Mem0Tools, etc. receive injections
    container.wire(packages=['bias_mitigation.mas', 'bias_mitigation.memory'])
    logger.info('✅ Dependency Injector container initialized and wired')
    setup_mlflow(tracking_uri, 'DSPy')
    logger.debug('MLflow setup completed.')

    _train_examples, dev_examples, train_ds_legacy, dev_ds_legacy = load_and_track_splits(
        base_dir=str(dataset_dir),
        version='v1.0',
    )
    logger.debug('Datasets loaded and tracked.')

    os.environ['MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION'] = 'True'
    dspy.settings.configure(provide_traceback=True)
    if subset <= 0:
        subset = len(dev_examples)
    evaluated_examples = dev_examples[:subset]
    with single_parent_evaluation_run(run_name) as run:
        logger.debug('Entered evaluation run context.')
        mlflow.log_input(train_ds_legacy, context='training')
        mlflow.log_input(dev_ds_legacy, context='evaluation')

        logger.info(f'✅ Logged training dataset: {train_ds_legacy.name}')
        logger.info(f'✅ Logged dev dataset: {dev_ds_legacy.name}')

        program_build_started = time.monotonic()
        mas_program = container.mas_program()
        logger.info(
            '⏱️ Program build completed in '
            f'{time.monotonic() - program_build_started:.2f}s '
            f'(intervention={mas_config.intervention}, subset={subset})'
        )
        run_metadata = {
            'protocol': mas_config.protocol,
            'intervention': mas_config.intervention,
            'num_agents': mas_config.num_agents,
            'rounds': mas_config.rounds,
            'llm_models': ','.join(model.name for model in mas_config.agent_models),
            'split': 'dev',
            'seed': str(getattr(mas_config, 'seed', 'unknown')),
            'run_id': run.info.run_id,
        }
        evaluator_init_started = time.monotonic()
        mas_evaluator = MASEvaluator(
            devset=evaluated_examples,
            backend=EvaluatorBackend(evaluator_backend),
            run_metadata=run_metadata,
        )
        logger.info(
            '⏱️ Evaluator initialization completed in '
            f'{time.monotonic() - evaluator_init_started:.2f}s '
            f'(backend={evaluator_backend})'
        )
        evaluation_started = time.monotonic()
        result = mas_evaluator(mas_program)
        logger.info(
            f'⏱️ Evaluation execution completed in {time.monotonic() - evaluation_started:.2f}s'
        )
        memory_stats: dict[str, int] = {}
        if getattr(mas_program, 'memory_rm', None) and hasattr(
            mas_program.memory_rm, 'stats_snapshot'
        ):
            memory_stats = mas_program.memory_rm.stats_snapshot()
        safety_scan_status = run_optional_safety_scan(run_safety_scan)

        if (
            evaluator_backend == EvaluatorBackend.DETERMINISTIC.value
            and min_system_robustness is not None
        ):
            validation_thresholds = {
                'MAS_System_Robustness': MetricThreshold(
                    threshold=min_system_robustness,
                    greater_is_better=True,
                )
            }
            mlflow.validate_evaluation_results(
                validation_thresholds=validation_thresholds,
                candidate_result=EvaluationResult(
                    metrics=result.get('genai_metrics', {}),
                    artifacts={},
                ),
            )
            logger.info(
                f'✅ Quality gate passed: MAS_System_Robustness >= {min_system_robustness:.3f}'
            )

        mlflow.log_param('evaluator_backend', result.get('backend', evaluator_backend))
        mlflow.log_param('evaluation.failure_count', int(result.get('failure_count', 0)))
        mlflow.log_param('evaluation.processed_count', int(result.get('processed_count', 0)))
        mlflow.log_metrics({
            key: float(value)
            for key, value in result.get('genai_metrics', {}).items()
            if isinstance(value, (int, float))
        })

        if isinstance(result.get('detailed_results'), list) and result['detailed_results']:
            mlflow.log_dict(
                {'rows': result['detailed_results']},
                'evaluation/detailed_results.json',
            )

        if isinstance(result.get('failed_examples'), list) and result['failed_examples']:
            mlflow.log_dict(
                {'rows': result['failed_examples']},
                'evaluation/failed_examples.json',
            )
            logger.warning(
                '⚠️ Deterministic evaluation encountered '
                f'{result.get("failure_count", 0)} failed examples '
                f'and processed {result.get("processed_count", 0)} successfully.'
            )

        mlflow.log_dict(
            {
                'backend': result.get('backend', evaluator_backend),
                'system_robustness': result.get('system_robustness', 0.0),
                'run_metadata': run_metadata,
                'stratified_metrics': result.get('stratified_metrics', {}),
                'metric_keys': sorted(result.get('genai_metrics', {}).keys()),
                'safety_scan': safety_scan_status,
                'memory_stats': memory_stats,
                'failure_count': result.get('failure_count', 0),
                'processed_count': result.get('processed_count', 0),
            },
            'evaluation/summary.json',
        )

        logger.success('✅ Evaluation completed successfully')
        logger.info(f'Backend: {result.get("backend", evaluator_backend)}')
        logger.info(f'Final system robustness: {result.get("system_robustness", 0.0):.3f}')


if __name__ == '__main__':
    main()
