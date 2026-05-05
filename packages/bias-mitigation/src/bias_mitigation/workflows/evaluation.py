"""Declarative evaluation workflow services shared by CLI entrypoints."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import dspy
import mlflow
import polars as pl
from loguru import logger
from mlflow.models import EvaluationResult, MetricThreshold

from bias_mitigation.containers import Container
from bias_mitigation.data.dataset_tracker import load_and_track_splits
from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.data.splitters import select_stratified_subset
from bias_mitigation.mas.evaluation.streaming import (
    LocalStreamConfig,
    build_live_run_dir_name,
    initialize_local_stream_layout,
)
from bias_mitigation.mas.evaluator import EvaluatorBackend, MASEvaluator
from bias_mitigation.memory.errors import MemoryConfigurationError

from .contracts import RunContext
from .statechart import WorkflowRuntime


def _resolve_effective_parallel_threads(config: MASConfig) -> int | None:
    return max(1, config.evaluator_concurrency.max_evaluation_threads)


def _apply_runtime_native_thread_limits(config: MASConfig) -> None:
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
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.active_run():
        mlflow.end_run()
    exp_name = f'{experiment_prefix}_{datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")}'
    mlflow.set_experiment(exp_name)


def run_optional_safety_scan(enabled: bool) -> dict[str, str]:
    if not enabled:
        return {'enabled': 'false', 'status': 'skipped'}

    if importlib.util.find_spec('giskard') is None:
        logger.warning('⚠️ Safety scan requested but `giskard` is not installed.')
        return {'enabled': 'true', 'status': 'missing_dependency', 'dependency': 'giskard'}

    logger.info('✅ Safety scan hook available (`giskard` import succeeded).')
    return {'enabled': 'true', 'status': 'available'}


def _log_table_artifacts(rows: list[dict[str, Any]], base_name: str, artifact_root: str) -> None:
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
    sample_rows = cast(list[dict[str, Any]], result.get('sample_outcomes', []))
    turn_rows = cast(list[dict[str, Any]], result.get('agent_turns', []))
    stream_metric_rows = cast(list[dict[str, Any]], result.get('stream_metric_rows', []))
    stream_round_metric_rows = cast(list[dict[str, Any]], result.get('stream_round_metric_rows', []))
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
    mlflow.log_dict({'rows': stream_round_metric_rows}, f'{analysis_root}/stream_round_metric_rows.json')
    mlflow.log_dict({'rows': stream_failure_rows}, f'{analysis_root}/stream_failure_rows.json')

    _log_table_artifacts(sample_rows, 'sample_outcomes', analysis_root)
    _log_table_artifacts(turn_rows, 'agent_turns', analysis_root)
    _log_table_artifacts(stream_metric_rows, 'stream_metric_rows', analysis_root)
    _log_table_artifacts(stream_round_metric_rows, 'stream_round_metric_rows', analysis_root)
    _log_table_artifacts(stream_failure_rows, 'stream_failure_rows', analysis_root)


@dataclass(slots=True)
class EvaluationWorkflowRuntimeImpl:
    """Typed runtime implementation for declarative evaluation workflows."""

    def prepare(self, context: RunContext) -> RunContext:
        return _prepare_context(context)

    def build(self, context: RunContext) -> RunContext:
        return _build_components(context)

    def execute(self, context: RunContext) -> RunContext:
        return _execute_evaluation(context)

    def persist(self, context: RunContext) -> RunContext:
        return _persist_outputs(context)

    def fail(self, context: RunContext) -> RunContext:
        return _handle_failure(context)


def _prepare_context(context: RunContext) -> RunContext:
    request = context.request
    setup_mlflow(request.tracking_uri, 'DSPy')

    overrides = {'intervention': request.intervention} if request.intervention else {}
    paths_to_load = [str(request.config_path)]
    if request.memory_config:
        paths_to_load.append(str(request.memory_config))

    mas_config = MASConfig.load_merged(*paths_to_load, cli_overrides=overrides).unwrap()

    configured_memory = mas_config.memory_config
    memory_config_payload = (
        configured_memory.model_dump(mode='json', exclude_none=True)
        if configured_memory is not None
        else {}
    )

    container = Container()
    container.config.from_dict({
        'mas_config': mas_config.model_dump(mode='json', exclude_none=True),
        'memory_config': memory_config_payload,
        'protocol': mas_config.protocol,
    })
    container.wire(packages=['bias_mitigation.mas', 'bias_mitigation.memory'])

    train_examples, dev_examples, train_ds_input, dev_ds_input = load_and_track_splits(
        base_dir=str(request.dataset_dir),
        version='v1.0',
    )

    effective_subset = len(dev_examples) if request.subset <= 0 else min(request.subset, len(dev_examples))
    evaluated_examples = select_stratified_subset(
        examples=dev_examples,
        subset_size=effective_subset,
        seed=request.subset_seed,
    )

    context.mas_config = mas_config
    context.container = container
    context.train_examples = train_examples
    context.dev_examples = dev_examples
    context.evaluated_examples = evaluated_examples
    context.train_ds_input = train_ds_input
    context.dev_ds_input = dev_ds_input
    context.effective_subset = effective_subset
    return context


def _build_components(context: RunContext) -> RunContext:
    request = context.request
    if context.container is None:
        raise RuntimeError('Workflow context missing DI container at build step.')
    if context.mas_config is None:
        raise RuntimeError('Workflow context missing MASConfig at build step.')

    run = mlflow.start_run(run_name=request.run_name)
    context.active_run = run

    mlflow.log_input(cast(Any, context.train_ds_input), context='training')
    mlflow.log_input(cast(Any, context.dev_ds_input), context='evaluation')

    build_started = time.monotonic()
    try:
        context.mas_program = context.container.mas_program()
    except MemoryConfigurationError as error:
        raise RuntimeError(
            'Failed to initialize Mem0 dependencies. '
            f'{error}. Please check your memory configuration and credentials.'
        ) from error

    logger.info(
        '⏱️ Program build completed in '
        f'{time.monotonic() - build_started:.2f}s '
        f'(intervention={context.mas_config.intervention}, subset={context.effective_subset})'
    )

    model_names = [model.name for model in context.mas_config.agent_models]
    model_labels = [
        f'{model.agent_name}:{model.name}' for model in context.mas_config.agent_models
    ]
    agent_model_map = {
        model.agent_name: model.name
        for model in context.mas_config.agent_models
    }
    experiment = mlflow.get_experiment(run.info.experiment_id)
    experiment_name = experiment.name

    live_run_dir_name = build_live_run_dir_name(
        template=context.mas_config.analysis_live_dir_template,
        tokens={
            'run_name': request.run_name,
            'run_id': run.info.run_id,
            'run_id_short': run.info.run_id[:8],
            'intervention': context.mas_config.intervention,
            'protocol': context.mas_config.protocol,
            'backend': request.evaluator_backend.value,
            'experiment_id': run.info.experiment_id,
            'experiment_name': experiment_name,
            'started_at': context.started_at_utc,
            'subset_seed': request.subset_seed,
        },
        token_max_length=context.mas_config.analysis_live_slug_max_length,
    )

    context.run_metadata = {
        'protocol': context.mas_config.protocol,
        'intervention': context.mas_config.intervention,
        'run_name': request.run_name,
        'experiment_id': run.info.experiment_id,
        'experiment_name': experiment_name,
        'num_agents': str(context.mas_config.num_agents),
        'rounds': str(context.mas_config.rounds),
        'llm_models': ', '.join(model_labels),
        'model_names': ', '.join(model_names),
        'agent_model_map': json.dumps(agent_model_map, ensure_ascii=False, sort_keys=True),
        'split': 'dev',
        'seed': str(getattr(context.mas_config, 'seed', 'unknown')),
        'subset_seed': str(request.subset_seed),
        'run_id': run.info.run_id,
        'live_analysis_dir': live_run_dir_name,
    }

    local_stream_config = LocalStreamConfig(
        root_dir=context.mas_config.analysis_local_root,
        run_id=run.info.run_id,
        flush_every_events=context.mas_config.stream_flush_every_events,
        fsync=context.mas_config.stream_fsync,
        write_csv=context.mas_config.stream_live_csv,
        max_buffered_events=context.mas_config.stream_max_buffered_events,
        drop_events_on_backpressure=context.mas_config.stream_drop_events_on_backpressure,
        run_dir_name=live_run_dir_name,
        run_manifest={
            'run_name': request.run_name,
            'experiment_id': run.info.experiment_id,
            'experiment_name': experiment_name,
            'backend': request.evaluator_backend.value,
            'intervention': context.mas_config.intervention,
            'protocol': context.mas_config.protocol,
            'agent_model_map': agent_model_map,
            'subset': context.effective_subset,
            'subset_seed': request.subset_seed,
            'started_at_utc': context.started_at_utc,
        },
        write_manifest=context.mas_config.analysis_live_write_manifest,
        index_filename=context.mas_config.analysis_live_index_filename,
    )
    initialize_local_stream_layout(local_stream_config)

    context.evaluator = MASEvaluator(
        devset=context.evaluated_examples,
        backend=request.evaluator_backend,
        run_metadata=context.run_metadata,
        parallel_num_threads=_resolve_effective_parallel_threads(context.mas_config),
        parallel_max_errors=context.mas_config.evaluator_max_errors,
        parallel_disable_progress_bar=context.mas_config.evaluator_disable_progress_bar,
        deterministic_execution_mode=context.mas_config.evaluator_deterministic_execution_mode,
        deterministic_window_size=context.mas_config.evaluator_window_size,
        deterministic_task_timeout_seconds=context.mas_config.evaluator_task_timeout_seconds,
        max_evaluation_threads=context.mas_config.evaluator_concurrency.max_evaluation_threads,
        concurrency_enable_monitoring=context.mas_config.evaluator_concurrency.enable_monitoring,
        local_stream_config=local_stream_config,
    )
    return context


def _execute_evaluation(context: RunContext) -> RunContext:
    if context.evaluator is None:
        raise RuntimeError('Workflow context missing evaluator at execute step.')
    if context.mas_program is None:
        raise RuntimeError('Workflow context missing MAS program at execute step.')
    if context.mas_config is None:
        raise RuntimeError('Workflow context missing MAS config at execute step.')

    os.environ['MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION'] = 'True'
    _apply_runtime_native_thread_limits(context.mas_config)
    dspy.settings.configure(provide_traceback=True)

    eval_started = time.monotonic()
    try:
        context.result = context.evaluator(context.mas_program)
    except KeyboardInterrupt:
        logger.warning('Evaluation interrupted by user cancellation request.')
        raise
    finally:
        memory_orchestrator = getattr(context.mas_program, 'memory_orchestrator', None)
        if memory_orchestrator is not None and hasattr(memory_orchestrator, 'shutdown'):
            memory_orchestrator.shutdown(wait=False)
    logger.info(f'⏱️ Evaluation execution completed in {time.monotonic() - eval_started:.2f}s')

    context.safety_scan_status = run_optional_safety_scan(context.request.run_safety_scan)
    if getattr(context.mas_program, 'memory_tools', None) and hasattr(
        context.mas_program.memory_tools, 'stats_snapshot'
    ):
        context.memory_stats = context.mas_program.memory_tools.stats_snapshot()
    return context


def _persist_outputs(context: RunContext) -> RunContext:
    request = context.request
    result = context.result

    if (
        request.evaluator_backend == EvaluatorBackend.DETERMINISTIC
        and request.min_system_robustness is not None
    ):
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
                metrics=result.get('genai_metrics', {}),
                artifacts={},
            ),
        )

    mlflow.log_param('evaluator_backend', result.get('backend', request.evaluator_backend.value))
    mlflow.log_param('evaluation.failure_count', int(result.get('failure_count', 0)))
    mlflow.log_param('evaluation.processed_count', int(result.get('processed_count', 0)))
    mlflow.log_metrics({
        key: float(value)
        for key, value in result.get('genai_metrics', {}).items()
        if isinstance(value, (int, float))
    })

    if isinstance(result.get('detailed_results'), list) and result['detailed_results']:
        mlflow.log_dict({'rows': result['detailed_results']}, 'evaluation/detailed_results.json')

    if isinstance(result.get('failed_examples'), list) and result['failed_examples']:
        mlflow.log_dict({'rows': result['failed_examples']}, 'evaluation/failed_examples.json')

    analysis_root = (
        context.mas_config.analysis_artifact_root
        if context.mas_config
        else 'evaluation/analysis/v1'
    )
    _log_analysis_artifacts(result, context.run_metadata, analysis_root)

    mlflow.log_dict(
        {
            'backend': result.get('backend', request.evaluator_backend.value),
            'system_robustness': result.get('system_robustness', 0.0),
            'run_metadata': context.run_metadata,
            'stratified_metrics': result.get('stratified_metrics', {}),
            'metric_keys': sorted(result.get('genai_metrics', {}).keys()),
            'safety_scan': context.safety_scan_status,
            'memory_stats': context.memory_stats,
            'failure_count': result.get('failure_count', 0),
            'processed_count': result.get('processed_count', 0),
            'stream_summary': result.get('stream_summary', {}),
        },
        'evaluation/summary.json',
    )

    if mlflow.active_run():
        mlflow.end_run(status='FINISHED')
    return context


def _handle_failure(context: RunContext) -> RunContext:
    if mlflow.active_run():
        mlflow.log_dict(
            {
                'error': context.error or 'unknown workflow failure',
                'run_metadata': context.run_metadata,
                'backend': context.request.evaluator_backend.value,
            },
            'evaluation/error.json',
        )
        mlflow.end_run(status='FAILED')
    return context


def build_evaluation_workflow_runtime() -> WorkflowRuntime:
    """Build typed runtime for declarative evaluation orchestration."""
    return EvaluationWorkflowRuntimeImpl()
