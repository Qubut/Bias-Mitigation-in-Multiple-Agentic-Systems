r"""CLI entry point for the Bias-Mitigation MAS evaluation workflow.

This script is the user-facing wrapper that researchers invoke to evaluate a
multi-agent system (MAS) on the benchmark dataset for one of the project's
intervention arms (baseline, baseline + GEPA prompt optimization, Mem0g
memory, or Mem0g + GEPA). It is intentionally a thin shell: it parses CLI
options, builds a ``RunRequest``, and hands control to the declarative
``WorkflowMachine`` defined in ``bias_mitigation.workflows.evaluation``.

The module also performs three runtime concerns that have to happen before
any other library is imported:

* It disables telemetry for Mem0, PostHog, MLflow, Chroma, and friends, so
  that reproducibility runs do not phone home.
* It installs a two-stage ``SIGINT`` handler so the first ``Ctrl+C``
  requests a graceful cancellation (via an env-var flag the workflow
  polls) and a second ``Ctrl+C`` force-exits the process.
* It patches ``threading.Thread.start`` to record where unnamed
  ``Thread-*`` workers are spawned, dumping a periodic JSON snapshot to
  ``logs/thread_tracer_live.json``. This is a debugging aid for tracking
  down rogue background threads in long evaluation runs.

Typical usage::

    python -m bias_mitigation.scripts.evaluate \\
        --config-path configs/mas_config.yaml \\
        --dataset-dir datasets/splits \\
        --intervention mem0g_gepa \\
        --subset 1500

Run with ``--help`` for the full list of flags.
"""

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
    """Write a JSON snapshot of recorded ``Thread-*`` start callsites.

    The snapshot includes the top callsites by frequency and a sample
    stack for each, which is useful for diagnosing background thread
    proliferation during long evaluation runs.

    Args:
        reason: Free-form tag stored alongside the snapshot (typically
            ``"periodic"`` or ``"final"``) so successive dumps can be
            distinguished.
    """
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
    """Monkey-patch ``threading.Thread.start`` to record unnamed thread spawns.

    Each call that creates a default-named ``Thread-*`` is logged with the
    nearest user-code frame (excluding this script and the standard
    library's ``threading`` module). The data is aggregated in-memory and
    flushed to ``logs/thread_tracer_live.json`` after every event, which
    makes it easy to attribute mysterious thread growth to a specific
    library or callsite during an evaluation run.
    """
    attr_name = 'start'

    def patched_start(thread_self: threading.Thread, *args: object, **kwargs: object) -> object:
        """Patched ``Thread.start`` that records and forwards each invocation."""
        if thread_self.name.startswith('Thread-'):
            stack = traceback.extract_stack(limit=30)
            candidate = [
                frame
                for frame in stack
                if '/scripts/evaluate.py' not in frame.filename
                and 'threading.py' not in frame.filename
            ]
            source = candidate[-1] if candidate else stack[-1]
            key = f'{source.filename}:{source.lineno}:{source.name}'
            formatted_stack = [
                f'{frame.filename}:{frame.lineno}:{frame.name}' for frame in candidate
            ]
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
    """Restore the original ``threading.Thread.start`` after the run completes.

    Called from ``main``'s ``finally`` block so the patch is local to a
    single CLI invocation and never leaks into other processes that
    happen to import this module.
    """
    attr_name = 'start'
    setattr(threading.Thread, attr_name, cast(Any, _ORIGINAL_THREAD_START))


def _log_thread_tracer_summary() -> None:
    """Log the top recorded ``Thread-*`` callsites and emit a final snapshot.

    Intended to run once at the end of evaluation. The summary is written
    at WARNING level when any events were recorded so it stands out in
    interactive logs; if no unnamed threads were ever spawned the function
    just notes that fact at INFO level.
    """
    with _THREAD_TRACER_LOCK:
        if not _THREAD_TRACER_EVENTS:
            logger.info('[ThreadTracer]: no Thread-* starts recorded in this run.')
            return
        counts = Counter(_THREAD_TRACER_EVENTS)
    top = ', '.join(f'{source} x{count}' for source, count in counts.most_common(10))
    logger.warning(f'[ThreadTracer]: top Thread-* start callsites: {top}')
    _write_thread_tracer_snapshot(reason='final')


def _request_cancellation() -> None:
    """Signal in-flight workflow steps to wind down gracefully.

    Sets the ``BIAS_MITIGATION_CANCEL_REQUESTED`` environment variable,
    which long-running components in the evaluation pipeline poll between
    steps so they can stop at a safe checkpoint instead of being killed
    mid-write.
    """
    os.environ[_CANCEL_ENV_VAR] = '1'


def _force_exit_handler(sig, frame) -> None:
    """Two-stage SIGINT handler: graceful cancel, then hard abort on repeat.

    The first ``Ctrl+C`` sets the cancellation flag and re-raises
    ``KeyboardInterrupt`` so Python's normal teardown still runs (giving
    the workflow a chance to flush partial results to disk). A second
    ``Ctrl+C`` calls ``os._exit(130)`` to terminate the process
    immediately, bypassing any hung threads that would otherwise prevent
    a clean exit.

    Args:
        sig: Unused; required by the ``signal.signal`` interface.
        frame: Unused; required by the ``signal.signal`` interface.

    Raises:
        KeyboardInterrupt: On the first interrupt, to let normal Python
            shutdown proceed.
    """
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
@click.option(
    '--tracking-uri',
    type=str,
    default=lambda: f'http://127.0.0.1:{os.environ.get("MLFLOW_PORT", "5000")}',
)
@click.option('--run-name', type=str, default='MAS_Experiment_Cooperative')
@click.option(
    '--intervention',
    type=click.Choice(['baseline', 'baseline_opt', 'mem0g', 'mem0g_gepa']),
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
@click.option(
    '--optimized-program',
    'optimized_program_path',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to a saved DSPy program JSON (e.g. from GEPA) to load before evaluation.',
)
@click.option(
    '--resume-from',
    'resume_from',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        'Path to stream_metric_rows.jsonl from an interrupted evaluation run. '
        'Already-completed example indices are skipped; new results are appended '
        'to the existing CSVs/JSONLs in the same live directory.'
    ),
)
def main(
    config_path: Path,
    dataset_dir: Path,
    tracking_uri: str,
    run_name: str,
    intervention: str | None,
    memory_config: Path | None,
    min_system_robustness: float | None,
    run_safety_scan: bool,
    subset: int,
    subset_seed: int,
    optimized_program_path: Path | None,
    resume_from: Path | None,
) -> None:
    r"""Evaluate a multi-agent system on the fairness/accuracy benchmark.

    Loads the MAS configuration from ``--config-path``, optionally merges
    a memory-config overlay, selects an intervention arm (baseline,
    baseline + GEPA, Mem0g, or Mem0g + GEPA), and runs the full evaluation
    workflow against the dataset splits in ``--dataset-dir``. Results are
    logged to MLflow at ``--tracking-uri`` under ``--run-name``.

    The work is delegated to a declarative ``WorkflowMachine`` whose
    runtime is assembled from dependency-injected services
    (``build_evaluation_workflow_runtime``). On a clean run the final
    system robustness score is printed; on failure a ``ClickException``
    is raised with the underlying error message. Use ``--resume-from``
    to continue an interrupted run from its ``stream_metric_rows.jsonl``
    without re-evaluating completed examples.

    Args:
        config_path: Path to the MAS YAML config (defaults to
            ``configs/mas_config.yaml``).
        dataset_dir: Directory containing the prepared dataset splits.
        tracking_uri: MLflow tracking server URI.
        run_name: Human-readable label for the MLflow run.
        intervention: Optional override of the config's intervention arm.
        memory_config: Optional path to a memory-config YAML overlay.
        min_system_robustness: Optional quality-gate threshold; the run
            fails when the final score is below this value.
        run_safety_scan: Enable the optional Giskard safety-layer hook.
        subset: Number of dev examples to evaluate.
        subset_seed: Seed for the deterministic stratified subset.
        optimized_program_path: Optional saved DSPy program JSON to load
            instead of constructing a fresh one (e.g. a GEPA checkpoint).
        resume_from: Optional ``stream_metric_rows.jsonl`` from a prior
            interrupted run; completed examples are skipped and new
            results appended to the existing live directory.

    Raises:
        click.ClickException: When the workflow finishes with an error
            recorded in its final context.

    Example:
        Reproduce the strongest intervention arm with the default
        subset::

            python -m bias_mitigation.scripts.evaluate \\
                --intervention mem0g_gepa
    """
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
        min_system_robustness=min_system_robustness,
        run_safety_scan=run_safety_scan,
        subset=subset,
        subset_seed=subset_seed,
        optimized_program_path=optimized_program_path,
        resume_from=resume_from,
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
        logger.info(
            f'Final system robustness: {final_context.result.get("system_robustness", 0.0):.3f}'
        )
    finally:
        _log_thread_tracer_summary()
        _restore_thread_start_tracer()


if __name__ == '__main__':
    main()
