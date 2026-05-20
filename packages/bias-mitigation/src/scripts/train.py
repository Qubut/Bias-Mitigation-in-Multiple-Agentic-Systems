r"""CLI entry point for GEPA-based training of the bias-mitigation MAS.

This is the user-facing script researchers run to reproduce the
prompt-optimization experiments from the paper. It loads the MAS
configuration, prepares the training and validation splits, runs GEPA
on a ``MASProgram`` to optimize its shared prompts, optionally evaluates
the resulting program on a held-out stratified subset, and persists the
optimized DSPy program to disk for later evaluation.

Usage:
    Run as a module so package-relative imports resolve correctly::

        python -m bias_mitigation.scripts.train \\
            --config-path configs/mas_config.yaml \\
            --dataset-dir datasets/splits \\
            --intervention baseline_opt

    Pass ``--help`` for the full list of options. The most important
    flags are ``--intervention`` (which GEPA-flavoured strategy to
    optimize), ``--train-subset`` / ``--subset`` (dataset caps used for
    quick sanity runs), and ``--optimized-program-path`` (where to save
    the artifact).

Artifacts:
    On success the script writes the optimized DSPy program to
    ``MASConfig.gepa.save_path`` (or the path passed via
    ``--optimized-program-path``) and logs run-level metrics, GEPA stats,
    and the configuration to the active MLflow tracking server.

Architecture:
    The script is intentionally thin — it converts CLI options into a
    :class:`RunRequest`, builds the training workflow runtime, and lets
    the shared ``WorkflowMachine`` drive the prepare → build → execute →
    persist stages. It mirrors ``scripts/evaluate.py`` so the two
    pipelines stay aligned; only the ``execute`` stage differs (see
    ``workflows/training.py``).
"""

from __future__ import annotations

import logging
import os
import signal
import warnings
from pathlib import Path

from bias_mitigation.workflows import RunContext, RunMode, RunRequest, WorkflowMachine
from bias_mitigation.workflows.training import build_training_workflow_runtime

# Telemetry must be disabled before importing libraries that may initialize
# background clients at import time (PostHog, Chroma, etc.).
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
from tqdm import tqdm

# Route loguru output through tqdm.write so the GEPA progress bar is not
# overwritten by interleaved log lines.
logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=''),
    colorize=True,
    format=(
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - '
        '<level>{message}</level>\n'
    ),
)


warnings.filterwarnings(
    'ignore', message='There is no current event loop', category=DeprecationWarning
)
logging.getLogger('mlflow.utils.git_utils').setLevel(logging.ERROR)

_CANCEL_ENV_VAR = 'BIAS_MITIGATION_CANCEL_REQUESTED'
_INTERRUPT_STATE = {'count': 0}


def _force_exit_handler(sig, frame) -> None:
    """Handle SIGINT in two stages: graceful cancel, then hard abort.

    GEPA runs are long and expensive, so a single Ctrl+C is interpreted
    as a cooperative cancellation request (the workflow polls the
    ``BIAS_MITIGATION_CANCEL_REQUESTED`` env var between stages and tears
    down cleanly, preserving partial MLflow artifacts). A second Ctrl+C
    indicates the user wants out immediately and we call ``os._exit``
    with status code 130 to bypass any blocked shutdown logic.

    Args:
        sig: POSIX signal number (unused; accepted to match the
            ``signal.signal`` handler signature).
        frame: Current stack frame (unused).
    """
    del sig, frame
    _INTERRUPT_STATE['count'] += 1
    if _INTERRUPT_STATE['count'] == 1:
        os.environ[_CANCEL_ENV_VAR] = '1'
        logger.warning(
            '⚠️  Ctrl+C received: cancellation requested. Press Ctrl+C again to force abort.'
        )
        raise KeyboardInterrupt
    logger.warning('⚠️  Second Ctrl+C: forcing immediate exit (code=130).')
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
@click.option('--run-name', type=str, default='MAS_Train_GEPA')
@click.option(
    '--intervention',
    type=click.Choice(['baseline_opt', 'mem0g_gepa']),
    default='baseline_opt',
    show_default=True,
    help='Which GEPA-flavoured intervention to optimize. '
    'When omitted, the value from the config YAML is used. '
    '(baseline / mem0g do not invoke GEPA — they are evaluate-only.)',
)
@click.option(
    '--memory-config',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Optional memory config YAML merged after base config.',
)
@click.option(
    '--optimized-program-path',
    type=click.Path(path_type=Path),
    default=None,
    help='Override save path for the optimized DSPy program. Defaults to MASConfig.gepa.save_path.',
)
@click.option(
    '--skip-validation',
    is_flag=True,
    default=False,
    help='Skip the post-GEPA stratified validation step.',
)
@click.option(
    '--subset',
    type=int,
    default=-1,
    show_default=True,
    help='Post-GEPA validation set cap (-1 or 0 = use full dev set).',
)
@click.option(
    '--train-subset',
    type=int,
    default=-1,
    show_default=True,
    help='GEPA trainset cap (-1 or 0 = use full training split).',
)
@click.option(
    '--valset-size',
    type=int,
    default=None,
    show_default=True,
    help='Override GepaConfig.valset_size (GEPA internal Pareto-tracker set size).',
)
@click.option(
    '--subset-seed',
    type=int,
    default=42,
    show_default=True,
    help='Seed for stratified validation subset selection.',
)
def main(
    config_path: Path,
    dataset_dir: Path,
    tracking_uri: str,
    run_name: str,
    intervention: str,
    memory_config: Path | None,
    optimized_program_path: Path | None,
    skip_validation: bool,
    subset: int,
    subset_seed: int,
    train_subset: int,
    valset_size: int | None,
) -> None:
    """Optimize the bias-mitigation MAS prompts with GEPA and save the result.

    Loads ``--config-path``, runs the GEPA prompt-optimization workflow on
    the training split under ``--dataset-dir``, optionally validates the
    optimized program on a stratified hold-out subset, and writes the
    final DSPy program plus MLflow metrics/artifacts to the configured
    tracking server.

    Use ``--intervention`` to choose which GEPA-flavoured strategy to
    optimize (``baseline_opt`` for the plain MAS, ``mem0g_gepa`` for the
    memory-augmented variant). Use ``--skip-validation`` together with
    ``--train-subset`` for fast smoke tests.
    """
    os.environ.pop(_CANCEL_ENV_VAR, None)
    _INTERRUPT_STATE['count'] = 0
    logger.info('Starting declarative training workflow...')
    signal.signal(signal.SIGINT, _force_exit_handler)

    request = RunRequest(
        mode=RunMode.TRAIN,
        config_path=config_path,
        dataset_dir=dataset_dir,
        tracking_uri=tracking_uri,
        run_name=run_name,
        intervention=intervention,
        memory_config=memory_config,
        min_system_robustness=None,
        run_safety_scan=False,
        subset=subset,
        subset_seed=subset_seed,
        optimized_program_path=optimized_program_path,
        skip_validation=skip_validation,
        train_subset=train_subset,
        valset_size_override=valset_size,
    )

    machine = WorkflowMachine(
        context=RunContext(request=request),
        runtime=build_training_workflow_runtime(),
    )
    final_context = machine.run()

    if final_context.error:
        raise click.ClickException(final_context.error)

    logger.success('✅ Training completed successfully')
    if final_context.optimized_program_path:
        logger.info(f'Optimized program → {final_context.optimized_program_path}')
    if final_context.gepa_stats:
        logger.info(f'GEPA stats: {final_context.gepa_stats}')
    if final_context.result.get('system_robustness') is not None:
        logger.info(
            f'Validation system robustness: '
            f'{final_context.result.get("system_robustness", 0.0):.3f}'
        )


if __name__ == '__main__':
    main()
