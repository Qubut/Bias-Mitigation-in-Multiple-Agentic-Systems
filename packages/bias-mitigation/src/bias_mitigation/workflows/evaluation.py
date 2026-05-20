"""Evaluation pipeline implementation for the bias-mitigation MAS.

This module realises the Pipeline pattern for the evaluation flow used by
the ``scripts/evaluate.py`` entrypoint and by the post-GEPA validation step
of ``scripts/train.py``. Each stage in the canonical sequence
``prepare`` -> ``build`` -> ``execute`` -> ``persist`` (with ``fail`` as the
recovery sink) is implemented as a small free function and exposed through
``EvaluationWorkflowRuntimeImpl``, which the ``WorkflowMachine`` state chart
drives.

Stage responsibilities at a glance:

* ``prepare``: configures MLflow, merges the ``MASConfig`` from YAML, wires
  the DI ``Container``, loads dataset splits, selects a stratified subset,
  and applies resume bookkeeping when an interrupted run is being continued.
* ``build``: starts the MLflow run, enables DSPy/MLflow tracing, constructs
  the MAS program and the ``MASEvaluator``, and initialises the on-disk
  streaming layout for live metric rows.
* ``execute``: runs the evaluator over the prepared examples, capturing
  fairness/accuracy results and tearing down memory orchestrator threads.
* ``persist``: validates configured robustness thresholds and best-effort
  logs all parameters, metrics, summary JSON, and tabular analysis
  artefacts to MLflow, then closes the run.
* ``fail``: writes an ``error.json`` artefact and closes the active MLflow
  run as ``FAILED``.

The pipeline is intentionally side-effect heavy (MLflow, filesystem,
optional Giskard scan). All MLflow calls during persistence go through a
Railway-Oriented Programming helper so a single artefact upload failure
cannot abort the rest of the persist step or leave the UI showing an empty
run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import dspy
import mlflow
import polars as pl
from loguru import logger
from mlflow.models import EvaluationResult, MetricThreshold
from returns.io import impure_safe
from returns.maybe import Maybe
from returns.pipeline import is_successful
from returns.result import safe
from returns.unsafe import unsafe_perform_io

from bias_mitigation.containers import Container
from bias_mitigation.data.dataset_tracker import load_and_track_splits
from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.data.splitters import select_stratified_subset
from bias_mitigation.mas.evaluation.streaming import (
    LocalStreamConfig,
    build_live_run_dir_name,
    initialize_local_stream_layout,
)
from bias_mitigation.mas.evaluator import MASEvaluator
from bias_mitigation.memory.errors import MemoryConfigurationError

from .contracts import RunContext, RunMode
from .statechart import WorkflowRuntime


def _resolve_effective_parallel_threads(config: MASConfig) -> int | None:
    """Return the number of evaluator worker threads to request from DSPy.

    Guarantees a floor of 1 so misconfigured YAML cannot accidentally
    serialize the evaluator into a no-op with zero workers.

    Args:
        config: Loaded MAS configuration whose ``evaluator_concurrency``
            section governs runtime parallelism.

    Returns:
        The clamped thread count (always ``>= 1``).
    """
    return max(1, config.evaluator_concurrency.max_evaluation_threads)


def _apply_runtime_native_thread_limits(config: MASConfig) -> None:
    """Cap BLAS/OpenMP thread pools to prevent oversubscription during eval.

    DSPy spawns its own worker pool; if NumPy / scikit-learn / PyTorch
    backends in turn launch a full OpenMP team per worker the host quickly
    thrashes. We therefore set a conservative cap on the common native
    thread environment variables, but only when the user has not set them
    explicitly. Memory-graph interventions (``mem0g``/``mem0g_gepa``) are
    more sensitive to context switching and get an even tighter cap.

    Args:
        config: MAS configuration whose ``intervention`` field determines
            the cap.

    Side Effects:
        Mutates ``os.environ`` for any uncapped thread variables and logs
        which ones were touched at INFO level.
    """
    intervention = str(config.intervention)
    native_thread_cap = 1 if intervention in {'mem0g', 'mem0g_gepa'} else 2
    native_thread_env_vars = (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    )
    updated_vars: list[str] = []
    for env_var in native_thread_env_vars:
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(native_thread_cap)
            updated_vars.append(env_var)

    if updated_vars:
        logger.info(
            'Applied runtime native thread limits '
            f'(cap={native_thread_cap}, vars={", ".join(updated_vars)}).'
        )


def setup_mlflow(tracking_uri: str, experiment_prefix: str) -> None:
    """Point MLflow at the configured tracking server and create an experiment.

    Idempotent across re-invocations: any lingering active run is closed
    first, so accidental re-entry (e.g. from a notebook) does not orphan a
    run. The experiment name is suffixed with a UTC timestamp to keep
    re-runs from colliding in the MLflow UI.

    Args:
        tracking_uri: The MLflow tracking URI (file://, http://, sqlite://).
        experiment_prefix: Prefix for the experiment name; a UTC timestamp
            is appended to form the actual experiment identifier.

    Side Effects:
        Sets the process-global MLflow tracking URI and active experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    Maybe.from_optional(mlflow.active_run()).map(lambda _: mlflow.end_run())
    exp_name = f'{experiment_prefix}_{datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")}'
    mlflow.set_experiment(exp_name)


def run_optional_safety_scan(enabled: bool) -> dict[str, str]:
    """Probe whether a Giskard-based safety scan can be performed.

    Giskard is an optional dependency in this project. Rather than failing
    a research run when the user did not install it, this helper reports a
    structured status that ``persist`` logs as part of the run summary.

    Args:
        enabled: Caller intent, typically ``RunRequest.run_safety_scan``.

    Returns:
        A status dict with at minimum ``enabled`` and ``status`` keys.
        ``status`` is one of ``'skipped'`` (not requested),
        ``'missing_dependency'`` (requested but ``giskard`` is not
        importable), or ``'available'`` (ready to run).
    """
    if not enabled:
        return {'enabled': 'false', 'status': 'skipped'}
    # Maybe.value_or(None) collapses the functor: None ⇒ giskard absent, spec ⇒ present.
    giskard_spec = Maybe.from_optional(importlib.util.find_spec('giskard')).value_or(None)
    if giskard_spec is None:
        logger.warning('⚠️ Safety scan requested but `giskard` is not installed.')
        return {'enabled': 'true', 'status': 'missing_dependency', 'dependency': 'giskard'}
    logger.info('✅ Safety scan hook available (`giskard` import succeeded).')
    return {'enabled': 'true', 'status': 'available'}


def _log_table_artifacts(rows: list[dict[str, Any]], base_name: str, artifact_root: str) -> None:
    """Materialise *rows* as CSV and Parquet artefacts and upload to MLflow.

    Two formats are written so downstream analysis can either be quick
    (CSV, openable in any spreadsheet) or efficient (Parquet, columnar).
    The function is a no-op when ``rows`` is empty, keeping MLflow runs
    free of zero-byte artefacts.

    Args:
        rows: List of dict rows; nested dicts/lists are stringified by
            ``_to_tabular_rows`` to keep the schema flat.
        base_name: Stem used for both files (``<base>.csv``, ``<base>.parquet``).
        artifact_root: MLflow artifact subdirectory to upload into.
    """
    if not rows:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame = pl.DataFrame(_to_tabular_rows(rows))
        csv_path = Path(tmp_dir) / f'{base_name}.csv'
        parquet_path = Path(tmp_dir) / f'{base_name}.parquet'

        frame.write_csv(csv_path)
        frame.write_parquet(parquet_path)
        mlflow.log_artifact(str(csv_path), artifact_path=artifact_root)
        mlflow.log_artifact(str(parquet_path), artifact_path=artifact_root)


def _to_tabular_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested values into strings so Polars can infer a uniform schema.

    Mixed-type columns (e.g. some rows containing a ``dict`` of per-agent
    scores and others containing ``None``) cause Polars to raise during
    ``DataFrame`` construction. Stringifying complex values is a pragmatic
    workaround that preserves information for downstream analysis without
    forcing a richer column model.

    Args:
        rows: Raw rows from the evaluator output.

    Returns:
        New list of dicts with the same keys but with ``dict`` / ``list``
        values converted to their ``str`` representation.
    """

    def normalize(value: Any) -> Any:
        match value:
            case dict() | list():
                return str(value)
            case _:
                return value

    return [{key: normalize(value) for key, value in row.items()} for row in rows]


def _log_analysis_artifacts(
    result: dict[str, Any],
    run_metadata: dict[str, Any],
    analysis_root: str,
) -> None:
    """Persist the rich per-example / per-round analysis layer to MLflow.

    Beyond the headline genai metrics, the evaluator emits several row-level
    datasets that downstream notebooks rely on (sample outcomes, agent
    turns, streaming metric / round / failure rows). This helper writes both
    a normalised JSON view and the tabular CSV/Parquet artefacts, plus a
    schema descriptor that captures counts and generation metadata.

    Args:
        result: Raw evaluator output dictionary.
        run_metadata: Run-identifying metadata copied into the schema file.
        analysis_root: MLflow artefact subdirectory under which all files
            are placed (typically ``evaluation/analysis/v1``).
    """
    sample_rows = cast(list[dict[str, Any]], result.get('sample_outcomes', []))
    turn_rows = cast(list[dict[str, Any]], result.get('agent_turns', []))
    stream_metric_rows = cast(list[dict[str, Any]], result.get('stream_metric_rows', []))
    stream_round_metric_rows = cast(
        list[dict[str, Any]], result.get('stream_round_metric_rows', [])
    )
    stream_failure_rows = cast(list[dict[str, Any]], result.get('stream_failure_rows', []))
    schema = cast(dict[str, Any], result.get('analysis_schema', {}))

    mlflow.log_dict(
        {
            **schema,
            'run_metadata': run_metadata,
            'generated_at_utc': datetime.now(tz=UTC).isoformat(),
            'sample_outcomes_count': len(sample_rows),
            'agent_turns_count': len(turn_rows),
            'stream_metric_rows_count': len(stream_metric_rows),
            'stream_round_metric_rows_count': len(stream_round_metric_rows),
            'stream_failure_rows_count': len(stream_failure_rows),
        },
        f'{analysis_root}/schema.json',
    )
    mlflow.log_dict({'rows': sample_rows}, f'{analysis_root}/sample_outcomes.json')
    mlflow.log_dict({'rows': turn_rows}, f'{analysis_root}/agent_turns.json')
    mlflow.log_dict({'rows': stream_metric_rows}, f'{analysis_root}/stream_metric_rows.json')
    mlflow.log_dict(
        {'rows': stream_round_metric_rows}, f'{analysis_root}/stream_round_metric_rows.json'
    )
    mlflow.log_dict({'rows': stream_failure_rows}, f'{analysis_root}/stream_failure_rows.json')

    _log_table_artifacts(sample_rows, 'sample_outcomes', analysis_root)
    _log_table_artifacts(turn_rows, 'agent_turns', analysis_root)
    _log_table_artifacts(stream_metric_rows, 'stream_metric_rows', analysis_root)
    _log_table_artifacts(stream_round_metric_rows, 'stream_round_metric_rows', analysis_root)
    _log_table_artifacts(stream_failure_rows, 'stream_failure_rows', analysis_root)


def _try_log(label: str, fn: Callable[[], Any]) -> None:
    """Run *fn* and swallow its exception, logging the failure with context.

    The persist stage performs many independent MLflow calls (params,
    metrics, dict artefacts, table uploads). A single failure -- e.g. a
    transient network blip when pushing to a remote tracking store --
    should not blank out the entire run from the UI. Wrapping each call
    via ``returns.result.safe`` converts exceptions into a ``Result``
    container that this helper inspects and logs.

    This explicit guarding is necessary because the surrounding state
    machine uses ``catch_errors_as_events=True``, which would otherwise
    silently consume the exception.

    Args:
        label: Human-readable tag included in the failure log for easy
            grep-ability when triaging runs.
        fn: Zero-argument thunk performing the MLflow side effect.
    """
    result = safe(fn)()
    if not is_successful(result):
        logger.exception(f'[persist]: MLflow {label} failed: {result.failure()}')


@dataclass(slots=True)
class EvaluationWorkflowRuntimeImpl:
    """Concrete ``WorkflowRuntime`` for evaluation-only runs.

    Methods on this class are thin adapters that forward to the module-level
    stage functions. Keeping the heavy lifting in free functions makes each
    stage trivially unit-testable in isolation, while the class itself
    satisfies the ``WorkflowRuntime`` protocol expected by
    ``WorkflowMachine``.
    """

    def prepare(self, context: RunContext) -> RunContext:
        """Load config, wire DI, fetch dataset splits, and pick the eval subset.

        Args:
            context: Run context with at least ``request`` populated.

        Returns:
            The same context, populated with ``mas_config``, ``container``,
            ``train_examples``, ``dev_examples``, ``evaluated_examples``,
            and dataset input handles.
        """
        return _prepare_context(context)

    def build(self, context: RunContext) -> RunContext:
        """Start the MLflow run and construct the MAS program and evaluator.

        Args:
            context: Context produced by ``prepare``.

        Returns:
            The same context, populated with ``active_run``, ``mas_program``,
            ``evaluator``, ``run_metadata``, and the on-disk streaming layout.
        """
        return _build_components(context)

    def execute(self, context: RunContext) -> RunContext:
        """Run the evaluator over the prepared examples.

        Args:
            context: Context produced by ``build``.

        Returns:
            The same context, with ``result``, ``memory_stats``, and
            ``safety_scan_status`` filled in.
        """
        return _execute_evaluation(context)

    def persist(self, context: RunContext) -> RunContext:
        """Log all metrics, artefacts, and summary JSON to MLflow.

        Args:
            context: Context produced by ``execute``.

        Returns:
            The same context, with the MLflow run closed as ``FINISHED``.
        """
        return _persist_outputs(context)

    def fail(self, context: RunContext) -> RunContext:
        """Record the failure to MLflow and close the run as ``FAILED``.

        Args:
            context: Context whose ``error`` field describes what went wrong.

        Returns:
            The same context, unchanged apart from the closed MLflow run.
        """
        return _handle_failure(context)


def _read_completed_indices(jsonl_path: Path) -> set[int]:
    """Read the set of ``example_index`` values already streamed to disk.

    The evaluator continuously appends metric rows to
    ``stream_metric_rows.jsonl`` during a run. When a run is interrupted and
    resumed via ``RunRequest.resume_from``, we replay the file to discover
    which examples completed successfully and skip them on the next attempt.

    Malformed lines (typically the last row when the previous process was
    killed mid-write) are ignored so the corresponding example is simply
    re-evaluated rather than skipped silently.

    Args:
        jsonl_path: Path to a ``stream_metric_rows.jsonl`` file from a
            prior run.

    Returns:
        The set of integer example indices already completed. Empty if the
        file cannot be read.
    """
    try:
        return set(
            pl
            .read_ndjson(jsonl_path, ignore_errors=True)
            .filter(pl.col('example_index').is_not_null())
            .get_column('example_index')
            .cast(pl.Int64)
            .to_list()
        )
    except OSError:
        logger.warning(f'[resume]: could not read {jsonl_path} — starting from scratch.')
    return set()


def _prepare_context(context: RunContext) -> RunContext:
    """Implementation of the ``prepare`` pipeline stage.

    Args:
        context: Run context with a fully populated ``request``.

    Returns:
        The mutated context, ready for ``build``.
    """
    request = context.request
    setup_mlflow(request.tracking_uri, 'DSPy')

    overrides = {'intervention': request.intervention} if request.intervention else {}
    paths_to_load = [str(request.config_path)]
    if request.memory_config:
        paths_to_load.append(str(request.memory_config))

    mas_config = MASConfig.load_merged(*paths_to_load, cli_overrides=overrides).unwrap()

    container = Container(mas_config=mas_config)
    container.wire(packages=['bias_mitigation.mas', 'bias_mitigation.memory'])

    train_examples, dev_examples, train_ds_input, dev_ds_input = load_and_track_splits(
        base_dir=str(request.dataset_dir),
        version='v1.0',
    )

    effective_subset = (
        len(dev_examples) if request.subset <= 0 else min(request.subset, len(dev_examples))
    )
    evaluated_examples = select_stratified_subset(
        examples=dev_examples,
        subset_size=effective_subset,
        seed=request.subset_seed,
    )

    context.mas_config = mas_config
    context.container = container

    context.train_examples = train_examples
    context.dev_examples = dev_examples

    match request.mode:
        case RunMode.TRAIN:
            # Cap the trainset if requested.
            if request.train_subset > 0:
                context.train_examples = select_stratified_subset(
                    examples=train_examples,
                    subset_size=min(request.train_subset, len(train_examples)),
                    seed=request.subset_seed,
                )

            # Override valset size from CLI, if given.
            if request.valset_size_override is not None:
                mas_config.gepa.valset_size = request.valset_size_override

            # Build a non-overlapping 3-way split to prevent hold-out leakage.
            gepa_cfg = mas_config.gepa
            gepa_val_size = max(0, gepa_cfg.valset_size)
            holdout_size = max(0, gepa_cfg.validation_subset)
            shuffled_dev = select_stratified_subset(
                examples=dev_examples,
                subset_size=min(gepa_val_size + holdout_size, len(dev_examples)),
                seed=request.subset_seed,
            )
            context.gepa_val_examples = shuffled_dev[:gepa_val_size]
            context.holdout_examples = shuffled_dev[gepa_val_size : gepa_val_size + holdout_size]

        case _:
            pass

    if request.resume_from is not None:
        completed_indices = _read_completed_indices(request.resume_from)
        original_evaluated = evaluated_examples
        evaluated_examples = [
            ex for i, ex in enumerate(original_evaluated) if i not in completed_indices
        ]
        context.example_index_offset = len(completed_indices)
        logger.info(
            f'[resume]: skipping {len(completed_indices)} already-completed examples; '
            f'{len(evaluated_examples)} remaining of {len(original_evaluated)} total.'
        )

    context.evaluated_examples = evaluated_examples
    context.train_ds_input = train_ds_input
    context.dev_ds_input = dev_ds_input
    context.effective_subset = effective_subset
    return context


def _build_components(context: RunContext) -> RunContext:
    """Implementation of the ``build`` pipeline stage.

    Starts the MLflow run, enables DSPy + MLflow autologging (including
    compile tracing in TRAIN mode so GEPA iterations are visible in the
    UI), instantiates the MAS DSPy program through the DI container, and
    optionally reloads a previously-saved optimized program. It then
    materialises the per-run ``run_metadata`` tag bag, prepares the
    on-disk live streaming layout (JSONL + optional CSV mirror), and
    constructs the ``MASEvaluator`` that ``execute`` will invoke.

    The live-run directory name is reused when resuming, so all rows for
    the same logical run end up in a single folder regardless of how many
    restarts were required.

    Args:
        context: Context produced by ``prepare``.

    Returns:
        The mutated context, ready for ``execute``.

    Raises:
        RuntimeError: If required upstream fields are missing, if Mem0
            initialisation fails, or if loading a saved optimized program
            from ``request.optimized_program_path`` fails.
    """
    request = context.request
    if context.container is None:
        raise RuntimeError('Workflow context missing DI container at build step.')
    if context.mas_config is None:
        raise RuntimeError('Workflow context missing MASConfig at build step.')
    mas_config = context.mas_config
    container = cast(Any, context.container)

    run = mlflow.start_run(run_name=request.run_name)
    context.active_run = run

    # Enable DSPy + MLflow tracing for this run.  Both ``mlflow.autolog`` and
    # ``mlflow.dspy.autolog`` are themselves idempotent (each call replaces the
    # global config), and the training pipeline re-asserts compile-tracing
    # explicitly when GEPA runs — so eval-then-train in one process upgrades
    # naturally without a hand-rolled latch.  MUST happen after ``start_run``
    # so traces are tied to this ``run_id``.
    try:
        is_training = context.request.mode == RunMode.TRAIN
        mlflow.autolog(log_traces=True, log_models=False, silent=True)
        mlflow.dspy.autolog(
            log_traces=True,
            log_traces_from_compile=is_training,
            log_traces_from_eval=True,
            log_compiles=False,
            log_evals=False,
            silent=True,
        )
        logger.info('[tracing]: MLflow + DSPy autolog enabled (compile_tracing=%s).', is_training)
    except Exception as exc:
        logger.exception(f'[tracing]: failed to initialize tracing: {exc}')

    mlflow.log_input(cast(Any, context.train_ds_input), context='training')
    mlflow.log_input(cast(Any, context.dev_ds_input), context='evaluation')

    build_started = time.monotonic()
    try:
        context.mas_program = container.mas_program()
    except MemoryConfigurationError as error:
        raise RuntimeError(
            'Failed to initialize Mem0 dependencies. '
            f'{error}. Please check your memory configuration and credentials.'
        ) from error

    # Load saved GEPA / optimized program weights if requested.
    if request.optimized_program_path is not None:
        opt_path = request.optimized_program_path
        logger.info(f'[build]: loading optimized program from {opt_path}')
        if context.mas_program is None:
            raise RuntimeError('mas_program is None — cannot load optimized weights into it.')
        try:
            context.mas_program.load(str(opt_path))
            logger.success(f'[build]: optimized program loaded ← {opt_path}')
        except Exception as exc:
            raise RuntimeError(f'Failed to load optimized program from {opt_path}: {exc}') from exc

    logger.info(
        '⏱️ Program build completed in '
        f'{time.monotonic() - build_started:.2f}s '
        f'(intervention={mas_config.intervention}, subset={context.effective_subset})'
    )

    model_names = [model.name for model in mas_config.agent_models]
    model_labels = [f'{model.agent_name}:{model.name}' for model in mas_config.agent_models]
    agent_model_map = {model.agent_name: model.name for model in mas_config.agent_models}
    experiment = mlflow.get_experiment(run.info.experiment_id)
    experiment_name = experiment.name

    # When resuming, re-use the original run's directory so new rows are
    # appended to the same JSONL/CSV files rather than scattered across a
    # second directory.
    if request.resume_from is not None:
        live_run_dir_name = request.resume_from.parent.name
        logger.info(f'[resume]: writing to existing run directory: {live_run_dir_name}')
    else:
        live_run_dir_name = build_live_run_dir_name(
            template=mas_config.analysis_live_dir_template,
            tokens={
                'run_name': request.run_name,
                'run_id': run.info.run_id,
                'run_id_short': run.info.run_id[:8],
                'intervention': mas_config.intervention,
                'protocol': mas_config.protocol,
                'experiment_id': run.info.experiment_id,
                'experiment_name': experiment_name,
                'started_at': context.started_at_utc,
                'subset_seed': request.subset_seed,
            },
            token_max_length=mas_config.analysis_live_slug_max_length,
        )

    context.run_metadata = {
        'protocol': mas_config.protocol,
        'intervention': mas_config.intervention,
        'run_name': request.run_name,
        'experiment_id': run.info.experiment_id,
        'experiment_name': experiment_name,
        'num_agents': str(mas_config.num_agents),
        'rounds': str(mas_config.rounds),
        'llm_models': ', '.join(model_labels),
        'model_names': ', '.join(model_names),
        'agent_model_map': json.dumps(agent_model_map, ensure_ascii=False, sort_keys=True),
        'split': 'dev',
        'seed': str(getattr(mas_config, 'seed', 'unknown')),
        'subset_seed': str(request.subset_seed),
        'run_id': run.info.run_id,
        'live_analysis_dir': live_run_dir_name,
    }

    local_stream_config = LocalStreamConfig(
        root_dir=mas_config.analysis_local_root,
        run_id=run.info.run_id,
        flush_every_events=mas_config.stream_flush_every_events,
        fsync=mas_config.stream_fsync,
        write_csv=mas_config.stream_live_csv,
        max_buffered_events=mas_config.stream_max_buffered_events,
        drop_events_on_backpressure=mas_config.stream_drop_events_on_backpressure,
        run_dir_name=live_run_dir_name,
        run_manifest={
            'run_name': request.run_name,
            'experiment_id': run.info.experiment_id,
            'experiment_name': experiment_name,
            'intervention': mas_config.intervention,
            'protocol': mas_config.protocol,
            'agent_model_map': agent_model_map,
            'subset': context.effective_subset,
            'subset_seed': request.subset_seed,
            'started_at_utc': context.started_at_utc,
        },
        write_manifest=mas_config.analysis_live_write_manifest,
        index_filename=mas_config.analysis_live_index_filename,
    )
    initialize_local_stream_layout(local_stream_config)

    context.evaluator = MASEvaluator(
        devset=context.evaluated_examples,
        run_metadata=context.run_metadata,
        parallel_num_threads=_resolve_effective_parallel_threads(mas_config),
        parallel_max_errors=mas_config.evaluator_max_errors,
        parallel_disable_progress_bar=mas_config.evaluator_disable_progress_bar,
        local_stream_config=local_stream_config,
        index_offset=context.example_index_offset,
    )
    return context


def _assert_execute_ready(context: RunContext) -> None:
    """Validate that ``build`` left the context in a runnable state.

    This is a defensive guard against partially-initialised contexts that
    could otherwise surface as confusing ``None``-attribute errors deep in
    the evaluator. By failing here we keep ``execute`` body free of
    ``None`` checks and let the type checker treat the relevant fields as
    non-optional after the call.

    Args:
        context: Context produced by ``build``.

    Raises:
        RuntimeError: If ``evaluator``, ``mas_program``, or ``mas_config``
            is missing.
    """
    if context.evaluator is None:
        raise RuntimeError('Workflow context missing evaluator at execute step.')
    if context.mas_program is None:
        raise RuntimeError('Workflow context missing MAS program at execute step.')
    if context.mas_config is None:
        raise RuntimeError('Workflow context missing MAS config at execute step.')


def _execute_evaluation(context: RunContext) -> RunContext:
    """Implementation of the ``execute`` pipeline stage.

    Invokes the ``MASEvaluator`` callable with the built DSPy program,
    using ``returns.io.impure_safe`` so exceptions are captured into an
    ``IOResult`` rather than propagated immediately. This gives us a single
    point at which to inspect failure and decide whether the state machine
    should transition to ``failed``. On success the raw evaluator output is
    stored on ``context.result``.

    A few important side concerns are handled here:

    * MLflow GenAI eval trace validation is skipped to avoid spurious
      false positives when DSPy traces contain custom shapes.
    * Native thread limits are applied before evaluation starts so they
      cover all worker processes.
    * Any background memory-orchestrator threads attached to the MAS
      program are shut down after evaluation so they do not leak into
      ``persist``.
    * Memory tool statistics and the optional safety scan status are
      captured for downstream persistence.

    Args:
        context: Context produced by ``build``.

    Returns:
        The mutated context with ``result``, ``memory_stats``, and
        ``safety_scan_status`` populated.

    Raises:
        Exception: Whatever the evaluator raised, re-raised after being
            unwrapped from the ``IOResult`` failure case.
    """
    _assert_execute_ready(context)
    # cast: guard above exhausted all None cases — all three are non-None here
    evaluator = cast(Any, context.evaluator)
    mas_program = cast(dspy.Module, context.mas_program)
    mas_config = cast(MASConfig, context.mas_config)

    os.environ['MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION'] = 'True'
    _apply_runtime_native_thread_limits(mas_config)
    dspy.settings.configure(provide_traceback=True)

    eval_started = time.monotonic()
    # ``impure_safe`` returns a callable typed as ``Any`` by the returns
    # stubs; binding the wrapper into an ``Any`` local lets mypy stop
    # flagging an "untyped call" at the point of invocation.
    safe_evaluator: Any = impure_safe(evaluator)
    io_result: Any = safe_evaluator(mas_program)
    eval_inner: Any = unsafe_perform_io(io_result)
    if is_successful(eval_inner):
        context.result = eval_inner.unwrap()
        logger.info(f'⏱️ Evaluation execution completed in {time.monotonic() - eval_started:.2f}s')
    else:
        raise eval_inner.failure()

    context.safety_scan_status = run_optional_safety_scan(context.request.run_safety_scan)
    memory_tools = getattr(mas_program, 'memory_tools', None)
    if memory_tools is not None and hasattr(memory_tools, 'stats_snapshot'):
        context.memory_stats = memory_tools.stats_snapshot()
    return context


def _persist_outputs(context: RunContext) -> RunContext:
    """Implementation of the ``persist`` pipeline stage.

    Logs every interesting facet of a run to MLflow so it can be reproduced
    and compared:

    * Optionally enforces a ``MAS_System_Robustness`` threshold when the
      deterministic evaluator backend is in use; failing this validation
      causes MLflow to raise.
    * Logs evaluator backend, failure counts, and processed counts as
      params.
    * Logs all numeric genai metrics in one batched call.
    * Uploads ``detailed_results`` and ``failed_examples`` as JSON
      artefacts when present.
    * Delegates the rich per-row analysis layer (sample outcomes, agent
      turns, stream rows) to ``_log_analysis_artifacts``.
    * Writes a final ``evaluation/summary.json`` that aggregates the most
      useful fields for quick run comparison.

    Every MLflow call is wrapped by ``_try_log`` so a single transient
    failure cannot leave the run silent in the UI.

    Args:
        context: Context produced by ``execute``.

    Returns:
        The same context with the MLflow run closed as ``FINISHED``.

    Raises:
        mlflow.exceptions.MlflowException: Only if
            ``request.min_system_robustness`` is set and the candidate run
            does not meet the threshold.
    """
    request = context.request
    result = context.result

    if request.min_system_robustness is not None:
        metric_threshold_cls: Any = MetricThreshold
        evaluation_result_cls: Any = EvaluationResult
        mlflow.validate_evaluation_results(
            validation_thresholds={
                'MAS_System_Robustness': metric_threshold_cls(
                    threshold=request.min_system_robustness,
                    greater_is_better=True,
                )
            },
            candidate_result=evaluation_result_cls(
                metrics=result.get('overall_metrics', {}),
                artifacts={},
            ),
        )

    # Each MLflow call is wrapped so a single failure (e.g. artifact upload to a
    # mis-configured store) doesn't take the whole persist step down silently.
    # Without this, `catch_errors_as_events=True` on the StateMachine swallows
    # the exception and the user sees an empty run in the MLflow UI.
    _try_log(
        'log_param(failure_count)',
        lambda: mlflow.log_param('evaluation.failure_count', int(result.get('failure_count', 0))),
    )
    _try_log(
        'log_param(processed_count)',
        lambda: mlflow.log_param(
            'evaluation.processed_count', int(result.get('processed_count', 0))
        ),
    )
    _try_log(
        'log_metrics(overall_metrics)',
        lambda: mlflow.log_metrics({
            key: float(value)
            for key, value in result.get('overall_metrics', {}).items()
            if isinstance(value, (int, float))
        }),
    )

    match result.get('detailed_results'):
        case [*_] as rows if rows:
            _try_log(
                'log_dict(detailed_results)',
                lambda: mlflow.log_dict({'rows': rows}, 'evaluation/detailed_results.json'),
            )

    match result.get('failed_examples'):
        case [*_] as rows if rows:
            _try_log(
                'log_dict(failed_examples)',
                lambda: mlflow.log_dict({'rows': rows}, 'evaluation/failed_examples.json'),
            )

    analysis_root = (
        Maybe
        .from_optional(context.mas_config)
        .map(lambda cfg: cfg.analysis_artifact_root)
        .value_or('evaluation/analysis/v1')
    )
    _try_log(
        'log_analysis_artifacts',
        lambda: _log_analysis_artifacts(result, context.run_metadata, analysis_root),
    )

    _try_log(
        'log_dict(summary)',
        lambda: mlflow.log_dict(
            {
                'system_robustness': result.get('system_robustness', 0.0),
                'run_metadata': context.run_metadata,
                'stratified_metrics': result.get('stratified_metrics', {}),
                'metric_keys': sorted(result.get('overall_metrics', {}).keys()),
                'safety_scan': context.safety_scan_status,
                'memory_stats': context.memory_stats,
                'failure_count': result.get('failure_count', 0),
                'processed_count': result.get('processed_count', 0),
                'stream_summary': result.get('stream_summary', {}),
            },
            'evaluation/summary.json',
        ),
    )

    Maybe.from_optional(mlflow.active_run()).map(lambda _: mlflow.end_run(status='FINISHED'))
    return context


def _handle_failure(context: RunContext) -> RunContext:
    """Implementation of the ``fail`` pipeline stage.

    Invoked by the state machine on any unrecoverable error from
    ``prepare`` / ``build`` / ``execute`` / ``persist``. Records the
    failure reason and the run-identifying metadata as
    ``evaluation/error.json`` so the run page in the MLflow UI carries
    enough context to triage the failure without re-running, then closes
    the active run as ``FAILED``. If no active MLflow run exists (e.g. the
    failure happened in ``prepare`` before ``build`` started the run),
    this function is a no-op.

    Args:
        context: Run context whose ``error`` field carries the diagnostic
            message.

    Returns:
        The context, unchanged apart from MLflow side effects.
    """

    def _log_and_close(_: Any) -> None:
        mlflow.log_dict(
            {
                'error': context.error or 'unknown workflow failure',
                'run_metadata': context.run_metadata,
            },
            'evaluation/error.json',
        )
        mlflow.end_run(status='FAILED')

    Maybe.from_optional(mlflow.active_run()).map(_log_and_close)
    return context


def build_evaluation_workflow_runtime() -> WorkflowRuntime:
    """Construct an evaluation runtime ready to be plugged into ``WorkflowMachine``.

    This factory exists so callers can stay decoupled from the concrete
    ``EvaluationWorkflowRuntimeImpl`` type and depend only on the
    ``WorkflowRuntime`` protocol.

    Returns:
        A fresh runtime instance whose stage methods implement the
        evaluation pipeline.

    Example:
        >>> runtime = build_evaluation_workflow_runtime()
        >>> machine = WorkflowMachine(runtime=runtime, context=RunContext(request))
        >>> machine.run()
    """
    return EvaluationWorkflowRuntimeImpl()
