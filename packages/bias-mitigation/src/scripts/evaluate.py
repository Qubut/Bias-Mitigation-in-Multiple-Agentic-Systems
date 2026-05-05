"""Packaged CLI entrypoint for declarative evaluation runs."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, cast

# Telemetry must be disabled before importing libraries that may initialize
# background clients at import time (e.g., PostHog/Chroma telemetry hooks).
os.environ['MEM0_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLED'] = 'true'
os.environ['DO_NOT_TRACK'] = 'true'
os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'true'
os.environ['CHROMADB_TELEMETRY_DISABLED'] = 'true'
if '/usr/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f'{os.environ.get("PATH", "")}:/usr/bin:/bin'

import click
from loguru import logger

from bias_mitigation.mas.evaluator import EvaluatorBackend
from bias_mitigation.workflows import RunContext, RunMode, RunRequest, WorkflowMachine
from bias_mitigation.workflows.evaluation import build_evaluation_workflow_runtime

warnings.filterwarnings(
    'ignore', message='There is no current event loop', category=DeprecationWarning
)
warnings.filterwarnings(
    'ignore',
    message=r"urllib3 \(.+\) or chardet \(.+\)/charset_normalizer \(.+\) doesn't match a supported version!",
    category=Warning,
)
logging.getLogger('mlflow.utils.git_utils').setLevel(logging.ERROR)

_CANCEL_ENV_VAR = 'BIAS_MITIGATION_CANCEL_REQUESTED'
_INTERRUPT_STATE = {'count': 0}
_INTERRUPT_LOCK = threading.Lock()
_THREAD_TRACER_LOCK = threading.Lock()
_THREAD_TRACER_EVENTS: list[str] = []
_THREAD_TRACER_SAMPLE_STACKS: dict[str, list[str]] = {}
_THREAD_TRACER_SNAPSHOT_PATH = Path('logs/thread_tracer_live.json')
_ORIGINAL_THREAD_START = threading.Thread.start


def _write_thread_tracer_snapshot(reason: str) -> None:
    with _THREAD_TRACER_LOCK:
        counts = Counter(_THREAD_TRACER_EVENTS)
        sample_stacks = {
            source: _THREAD_TRACER_SAMPLE_STACKS.get(source, [])
            for source, _ in counts.most_common(10)
        }
        payload = {
            'reason': reason,
            'event_count': len(_THREAD_TRACER_EVENTS),
            'top_sources': counts.most_common(20),
            'sample_stacks': sample_stacks,
        }
    _THREAD_TRACER_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _THREAD_TRACER_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _install_thread_start_tracer() -> None:
    attr_name = 'start'

    def patched_start(thread_self: threading.Thread, *args: object, **kwargs: object) -> object:
        if thread_self.name.startswith('Thread-'):
            stack = traceback.extract_stack(limit=30)
            candidate = [
                frame
                for frame in stack
                if '/scripts/evaluate.py' not in frame.filename and 'threading.py' not in frame.filename
            ]
            source = candidate[-1] if candidate else stack[-1]
            key = f'{source.filename}:{source.lineno}:{source.name}'
            formatted_stack = [f'{frame.filename}:{frame.lineno}:{frame.name}' for frame in candidate]
            with _THREAD_TRACER_LOCK:
                _THREAD_TRACER_EVENTS.append(key)
                if key not in _THREAD_TRACER_SAMPLE_STACKS:
                    _THREAD_TRACER_SAMPLE_STACKS[key] = formatted_stack
                event_count = len(_THREAD_TRACER_EVENTS)
            if event_count % 1 == 0:
                _write_thread_tracer_snapshot(reason='periodic')
        return _ORIGINAL_THREAD_START(thread_self, *args, **kwargs)

    setattr(threading.Thread, attr_name, cast(Any, patched_start))


def _restore_thread_start_tracer() -> None:
    attr_name = 'start'
    setattr(threading.Thread, attr_name, cast(Any, _ORIGINAL_THREAD_START))


def _log_thread_tracer_summary() -> None:
    with _THREAD_TRACER_LOCK:
        if not _THREAD_TRACER_EVENTS:
            logger.info('[ThreadTracer]: no Thread-* starts recorded in this run.')
            return
        counts = Counter(_THREAD_TRACER_EVENTS)
    top = ', '.join(f'{source} x{count}' for source, count in counts.most_common(10))
    logger.warning(f'[ThreadTracer]: top Thread-* start callsites: {top}')
    _write_thread_tracer_snapshot(reason='final')


def _request_cancellation() -> None:
    os.environ[_CANCEL_ENV_VAR] = '1'


def _force_exit_handler(sig, frame) -> None:
    """Two-stage SIGINT handler: graceful cancel, then hard abort on repeat."""
    del sig, frame
    with _INTERRUPT_LOCK:
        _INTERRUPT_STATE['count'] += 1
        interrupt_count = _INTERRUPT_STATE['count']

    if interrupt_count == 1:
        _request_cancellation()
        logger.warning(
            '⚠️  Ctrl+C received: cancellation requested. '
            'Finishing in-flight operation before shutdown; press Ctrl+C again to force abort.'
        )
        raise KeyboardInterrupt

    logger.warning('⚠️  Second Ctrl+C received: forcing immediate process exit (code=130).')
    os._exit(130)


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
    default=1500,
    show_default=True,
    help='Number of dev examples to evaluate (use small value for runtime validation).',
)
@click.option(
    '--subset-seed',
    type=int,
    default=42,
    show_default=True,
    help='Seed for deterministic stratified subset selection.',
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
    subset_seed: int,
) -> None:
    """Run evaluation via declarative workflow statechart + DI-backed services."""
    os.environ.pop(_CANCEL_ENV_VAR, None)
    _INTERRUPT_STATE['count'] = 0
    _install_thread_start_tracer()
    logger.info('Starting declarative evaluation workflow...')
    signal.signal(signal.SIGINT, _force_exit_handler)

    request = RunRequest(
        mode=RunMode.EVALUATE,
        config_path=config_path,
        dataset_dir=dataset_dir,
        tracking_uri=tracking_uri,
        run_name=run_name,
        intervention=intervention,
        memory_config=memory_config,
        evaluator_backend=EvaluatorBackend(evaluator_backend),
        min_system_robustness=min_system_robustness,
        run_safety_scan=run_safety_scan,
        subset=subset,
        subset_seed=subset_seed,
    )
    try:
        machine = WorkflowMachine(
            context=RunContext(request=request),
            runtime=build_evaluation_workflow_runtime(),
        )
        final_context = machine.run()

        if final_context.error:
            raise click.ClickException(final_context.error)

        logger.success('✅ Evaluation completed successfully')
        logger.info(f'Backend: {final_context.result.get("backend", evaluator_backend)}')
        logger.info(
            f'Final system robustness: {final_context.result.get("system_robustness", 0.0):.3f}'
        )
    finally:
        _log_thread_tracer_summary()
        _restore_thread_start_tracer()


if __name__ == '__main__':
    main()
