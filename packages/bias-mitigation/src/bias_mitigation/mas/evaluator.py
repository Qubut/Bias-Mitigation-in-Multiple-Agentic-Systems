"""Evaluation helpers for deterministic and GenAI MAS scoring backends."""

import asyncio
import json
import os
import re
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from hashlib import sha256
from typing import Any, ClassVar, cast

import dspy
from aiostream import stream
from loguru import logger
from mlflow.genai import evaluate
from returns.functions import raise_exception
from returns.future import future_safe
from returns.io import IOResult, impure_safe
from returns.pipeline import is_successful, managed, pipe
from returns.pointfree import bind, map_
from returns.result import Failure, Success, safe
from returns.unsafe import unsafe_perform_io
from tqdm import tqdm

from .evaluation.adapters import PredictFnAdapter
from .evaluation.aggregation import (
    aggregate_stratified_metric_rows,
    mean_metric_dict,
    metric_uncertainty,
    validate_stratified_dimensions,
)
from .evaluation.concurrency import ConcurrencyConfig, ConcurrencyManager
from .evaluation.models import (
    _AgentBiasSummary,
    _AgentTurnRow,
    _EvaluationMetadata,
    _EvaluatorOutput,
    _FailureExampleRow,
    _RoundMetricRow,
    _SampleOutcomeRow,
    _StratifiedRow,
)
from .evaluation.strategy import EvaluatorStrategyFactory
from .evaluation.streaming import (
    CsvFileMetricEventSink,
    EvaluationCompletedStreamEvent,
    InMemoryMetricEventSink,
    JsonlFileMetricEventSink,
    LocalStreamConfig,
    MetricStreamDispatcher,
    SampleFailureStreamEvent,
    SampleMetricsStreamEvent,
    SampleRoundMetricsStreamEvent,
)
from .evaluation.worker import (
    _DeterministicParallelWorker,
    _ExampleEvalRecord,
    _ParallelEvalTask,
    _ParallelEvalResult,
)
from .metrics import (
    amplification_rate,
    build_agent_turn_bias_series,
    build_round_bias_attribution,
    build_round_metric_series,
    emergence_rate,
    propagation_rate,
    system_robustness,
)


class EvaluatorBackend(StrEnum):
    """Available evaluator backend options."""

    DETERMINISTIC = 'deterministic'
    GENAI = 'genai'


class DeterministicExecutionMode(StrEnum):
    """Supported deterministic execution engines."""

    DSPY_BATCH = 'dspy_batch'
    SLIDING_WINDOW = 'sliding_window'


class MASEvaluator:
    """Run MAS evaluation with deterministic default and optional GenAI backend."""

    _MLFLOW_GENAI_MAX_WORKERS_ENV: ClassVar[str] = 'MLFLOW_GENAI_EVAL_MAX_WORKERS'
    _CANCEL_ENV_VAR: ClassVar[str] = 'BIAS_MITIGATION_CANCEL_REQUESTED'
    _DEFAULT_MAX_WORKERS_CAP: ClassVar[int] = 8
    _SLIDING_HEARTBEAT_SECONDS: ClassVar[float] = 20.0
    _THREAD_SNAPSHOT_TOP_K: ClassVar[int] = 10

    _METRIC_NAME_MAP: ClassVar[dict[str, str]] = {
        'system_robustness': 'MAS_System_Robustness',
        'emergence_rate': 'MAS_Emergence_Rate',
        'amplification_rate': 'MAS_Amplification_Rate',
        'propagation_rate': 'MAS_Propagation_Rate',
    }

    _STRATIFY_FIELDS: ClassVar[tuple[str, ...]] = (
        'dataset_name',
        'dataset_source',
        'stereoset_type',
        'category',
        'protocol',
        'llm_models',
        'intervention',
        'num_agents',
        'rounds',
        'split',
        'seed',
    )

    _BBQ_CATEGORIES: ClassVar[set[str]] = {
        'Age',
        'Disability_status',
        'Gender_identity',
        'Nationality',
        'Physical_appearance',
        'Race_ethnicity',
        'Race_x_gender',
        'Race_x_SES',
        'Religion',
        'SES',
        'Sexual_orientation',
    }

    _STEREOSET_CATEGORIES: ClassVar[set[str]] = {
        'intersentence',
        'intrasentence',
        'dev',
    }

    def __init__(
        self,
        devset: list[dspy.Example],
        backend: EvaluatorBackend | str = EvaluatorBackend.DETERMINISTIC,
        run_metadata: dict[str, Any] | None = None,
        parallel_num_threads: int | None = None,
        parallel_max_errors: int | None = None,
        parallel_disable_progress_bar: bool = False,
        deterministic_execution_mode: DeterministicExecutionMode
        | str = DeterministicExecutionMode.SLIDING_WINDOW,
        deterministic_window_size: int | None = None,
        deterministic_task_timeout_seconds: float = 180.0,
        max_evaluation_threads: int | None = None,
        concurrency_enable_monitoring: bool = True,
        local_stream_config: LocalStreamConfig | None = None,
    ):
        self.devset = devset
        self.backend = EvaluatorBackend(backend)
        self.run_metadata = run_metadata or {}
        self.parallel_num_threads = parallel_num_threads
        self.parallel_max_errors = parallel_max_errors
        self.parallel_disable_progress_bar = parallel_disable_progress_bar
        self.deterministic_execution_mode = DeterministicExecutionMode(deterministic_execution_mode)
        self.deterministic_window_size = deterministic_window_size
        self.deterministic_task_timeout_seconds = deterministic_task_timeout_seconds
        self.max_evaluation_threads = max_evaluation_threads
        self.concurrency_enable_monitoring = concurrency_enable_monitoring
        self.local_stream_config = local_stream_config

    @staticmethod
    def _parse_positive_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 1 else None

    @classmethod
    def _default_parallel_workers(cls) -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, cls._DEFAULT_MAX_WORKERS_CAP))

    def _resolve_parallel_workers(self) -> int:
        if self.parallel_num_threads is not None:
            return self.parallel_num_threads

        env_workers = self._parse_positive_int(os.getenv(self._MLFLOW_GENAI_MAX_WORKERS_ENV))
        if env_workers is not None:
            return env_workers

        return self._default_parallel_workers()

    def _resolve_max_evaluation_threads(self, parallel_workers: int) -> int:
        if self.max_evaluation_threads is None:
            return max(1, parallel_workers)
        return max(1, min(parallel_workers, self.max_evaluation_threads))

    def _build_concurrency_manager(self, parallel_workers: int) -> ConcurrencyManager:
        max_eval_threads = self._resolve_max_evaluation_threads(parallel_workers)
        return ConcurrencyManager(
            ConcurrencyConfig(
                max_evaluation_threads=max_eval_threads,
                enable_monitoring=self.concurrency_enable_monitoring,
            )
        )

    def is_cancel_requested(self) -> bool:
        return os.getenv(self._CANCEL_ENV_VAR, '0') == '1'

    @contextmanager
    def _with_mlflow_genai_max_workers(self, num_workers: int) -> Iterator[None]:
        previous_value = os.getenv(self._MLFLOW_GENAI_MAX_WORKERS_ENV)
        os.environ[self._MLFLOW_GENAI_MAX_WORKERS_ENV] = str(num_workers)
        try:
            yield
        finally:
            if previous_value is None:
                os.environ.pop(self._MLFLOW_GENAI_MAX_WORKERS_ENV, None)
            else:
                os.environ[self._MLFLOW_GENAI_MAX_WORKERS_ENV] = previous_value

    @staticmethod
    def _extract_system_robustness(metrics: dict[str, Any]) -> float:
        direct_key = 'MAS_System_Robustness'
        if direct_key in metrics:
            return float(metrics[direct_key])

        for key, value in metrics.items():
            if 'MAS_System_Robustness' in key:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        return 0.0

    def _to_mlflow_eval_dataset(self, devset: list[dspy.Example]) -> list[dict[str, Any]]:
        return [{'inputs': example.toDict()} for example in devset]

    @staticmethod
    def _feedback_value(feedback: Any) -> float:
        if hasattr(feedback, 'value'):
            return float(feedback.value)
        return float(feedback)

    def _extract_metadata(
        self,
        inputs: dict[str, Any],
        extra_metadata: dict[str, Any] | None,
    ) -> _EvaluationMetadata:
        merged: dict[str, Any] = {}
        merged.update(self.run_metadata)
        if extra_metadata:
            merged.update(extra_metadata)

        category = inputs.get('category') or merged.get('category') or 'unknown'
        dataset_source = inputs.get('source') or merged.get('dataset_source')
        stereoset_type = (
            inputs.get('original_type') or inputs.get('subcategory') or merged.get('stereoset_type')
        )

        dataset_name = (
            inputs.get('dataset_name')
            or inputs.get('dataset')
            or dataset_source
            or merged.get('dataset_name')
            or self._infer_dataset_name(str(category))
        )

        metadata_payload = {
            'dataset_name': str(dataset_name),
            'dataset_source': str(dataset_source or dataset_name),
            'stereoset_type': str(stereoset_type or 'none'),
            'category': str(category),
            'protocol': str(merged.get('protocol', 'unknown')),
            'llm_models': str(merged.get('llm_models', 'unknown')),
            'model_names': str(merged.get('model_names', 'unknown')),
            'intervention': str(merged.get('intervention', 'unknown')),
            'num_agents': str(merged.get('num_agents', 'unknown')),
            'rounds': str(merged.get('rounds', 'unknown')),
            'split': str(merged.get('split', 'unknown')),
            'seed': str(merged.get('seed', 'unknown')),
            'run_id': str(merged.get('run_id', 'unknown')),
            'agent_model_map': str(merged.get('agent_model_map', '{}')),
        }
        return _EvaluationMetadata.model_validate(metadata_payload)

    @staticmethod
    def _parse_agent_model_map(raw_map: str) -> dict[str, str]:
        return pipe(
            safe(json.loads)(raw_map),
            bind(lambda p: Success(p) if isinstance(p, dict) else Failure(TypeError('not a dict'))),
            map_(lambda p: {str(k): str(v) for k, v in p.items()}),
        ).value_or({})

    @staticmethod
    def _canonical_sample_text(inputs: dict[str, Any]) -> str:
        options = [
            str(inputs.get('ans0', '')),
            str(inputs.get('ans1', '')),
            str(inputs.get('ans2', '')),
        ]
        return '\n'.join([
            str(inputs.get('context', '')),
            str(inputs.get('question', '')),
            *options,
            str(inputs.get('category', 'unknown')),
        ])

    def _resolve_sample_id(self, inputs: dict[str, Any], example_index: int) -> str:
        direct_id = inputs.get('sample_id') or inputs.get('id')
        if isinstance(direct_id, str) and direct_id.strip():
            return direct_id.strip()
        if isinstance(direct_id, int):
            return str(direct_id)

        digest = sha256(self._canonical_sample_text(inputs).encode('utf-8')).hexdigest()[:16]
        return f'sample-{example_index}-{digest}'

    def _build_agent_turn_rows(
        self,
        inputs: dict[str, Any],
        prediction: dspy.Prediction,
        sample_id: str,
        example_index: int,
        metadata: _EvaluationMetadata,
        sample_run_id: str | None,
    ) -> list[_AgentTurnRow]:
        model_map = self._parse_agent_model_map(metadata.agent_model_map)
        base_turn_rows = build_agent_turn_bias_series(
            inputs=inputs,
            outputs=prediction,
            agent_model_map=model_map,
        )
        history = getattr(prediction, 'history', {})
        final_turn_index_by_agent = {
            str(agent_name): max(len(turns) - 1, 0) for agent_name, turns in history.items()
        }
        return _AgentTurnRow.from_series(
            base_turn_rows=base_turn_rows,
            final_turn_index_by_agent=final_turn_index_by_agent,
            sample_id=sample_id,
            sample_run_id=sample_run_id,
            example_index=example_index,
            metadata=metadata,
        )

    def _analysis_schema(self) -> dict[str, Any]:
        return {
            'artifact_schema_version': '1.1.0',
            'sample_outcomes_columns': _SampleOutcomeRow.analysis_columns(),
            'agent_turns_columns': _AgentTurnRow.analysis_columns(),
            'round_metrics_columns': _RoundMetricRow.analysis_columns(),
            'stratify_fields': list(self._STRATIFY_FIELDS),
        }

    @classmethod
    def _infer_dataset_name(cls, category: str) -> str:
        if category in cls._BBQ_CATEGORIES:
            return 'BBQ'
        if category in cls._STEREOSET_CATEGORIES:
            return 'StereoSet'
        return 'unknown'

    def _strata_key(self, record: _ExampleEvalRecord) -> tuple[str, ...]:
        return tuple(str(getattr(record.metadata, field)) for field in self._STRATIFY_FIELDS)

    def _validate_stratified_dimensions(self, rows: list[_StratifiedRow]) -> None:
        validate_stratified_dimensions(
            rows=[{'dimensions': row.dimensions} for row in rows],
            expected_fields=self._STRATIFY_FIELDS,
        )

    def _stratify_records(
        self,
        records: list[_ExampleEvalRecord],
    ) -> list[_StratifiedRow]:
        aggregate_rows = aggregate_stratified_metric_rows(
            records=[
                {
                    'metadata': record.metadata.model_dump(mode='python'),
                    'metrics': record.metrics,
                }
                for record in records
            ],
            stratify_fields=self._STRATIFY_FIELDS,
        )
        rows = [
            _StratifiedRow(
                dimensions=cast(dict[str, str], row['dimensions']),
                support=cast(int, row['support']),
                metrics=cast(dict[str, float], row['metrics']),
                ci95=cast(dict[str, float], row['ci95']),
            )
            for row in aggregate_rows
        ]
        self._validate_stratified_dimensions(rows)
        return rows

    def _evaluate_single_example(
        self,
        predict_fn: PredictFnAdapter,
        example: dspy.Example,
        extra_metadata: dict[str, Any] | None,
        example_index: int,
    ) -> _ExampleEvalRecord:
        inputs = example.toDict()
        sample_id = self._resolve_sample_id(inputs, example_index)
        prediction = predict_fn(**{**inputs, 'sample_id': sample_id})
        metadata = self._extract_metadata(inputs, extra_metadata)

        metric_extractors = {
            self._METRIC_NAME_MAP['system_robustness']: system_robustness,
            self._METRIC_NAME_MAP['emergence_rate']: emergence_rate,
            self._METRIC_NAME_MAP['amplification_rate']: amplification_rate,
            self._METRIC_NAME_MAP['propagation_rate']: propagation_rate,
        }
        sample_metrics = {
            metric_name: self._feedback_value(scorer_fn(inputs=inputs, outputs=prediction))
            for metric_name, scorer_fn in metric_extractors.items()
        }
        sample_run_id = cast(str | None, getattr(prediction, 'sample_run_id', None))
        agent_turns = self._build_agent_turn_rows(
            inputs=inputs,
            prediction=prediction,
            sample_id=sample_id,
            example_index=example_index,
            metadata=metadata,
            sample_run_id=sample_run_id,
        )
        bias_summary = _AgentBiasSummary.from_turn_rows(agent_turns)

        sample_outcome = _SampleOutcomeRow(
            sample_id=sample_id,
            example_index=example_index,
            mlflow_run_id=metadata.run_id,
            dataset_name=metadata.dataset_name,
            dataset_source=metadata.dataset_source,
            stereoset_type=metadata.stereoset_type,
            category=metadata.category,
            protocol=metadata.protocol,
            llm_models=metadata.llm_models,
            model_names=metadata.model_names,
            intervention=metadata.intervention,
            num_agents=metadata.num_agents,
            rounds=metadata.rounds,
            split=metadata.split,
            seed=metadata.seed,
            question_polarity=str(inputs.get('question_polarity', 'unknown')),
            context_condition=str(inputs.get('context_condition', 'unknown')),
            label=inputs.get('label'),
            gold_answer_text=str(
                [inputs.get('ans0', ''), inputs.get('ans1', ''), inputs.get('ans2', '')][
                    int(inputs.get('label', 0))
                ]
            )
            if isinstance(inputs.get('label'), int)
            else 'unknown',
            system_robustness=sample_metrics[self._METRIC_NAME_MAP['system_robustness']],
            emergence_rate=sample_metrics[self._METRIC_NAME_MAP['emergence_rate']],
            amplification_rate=sample_metrics[self._METRIC_NAME_MAP['amplification_rate']],
            propagation_rate=sample_metrics[self._METRIC_NAME_MAP['propagation_rate']],
            turn_count=max(
                (len(turns) for turns in getattr(prediction, 'history', {}).values()), default=0
            ),
            processed_flag=True,
            failure_reason=None,
            sample_run_id=sample_run_id,
            agent_model_map=metadata.agent_model_map,
            first_biased_turn_by_agent=bias_summary.first_biased_turn_by_agent,
            final_is_biased_by_agent=bias_summary.final_is_biased_by_agent,
            final_answers=getattr(prediction, 'final_answers', {}),
        )
        round_bias_rows = build_round_bias_attribution([
            turn.model_dump(mode='python') for turn in agent_turns
        ])
        round_bias_by_turn = {int(row['turn_index']): row for row in round_bias_rows}
        round_metric_rows = [
            _RoundMetricRow.from_components(
                sample_id=sample_id,
                example_index=example_index,
                metadata=metadata,
                round_metrics=round_metrics,
                bias_row=round_bias_by_turn.get(int(round_metrics['turn_index'])),
            )
            for round_metrics in build_round_metric_series(inputs=inputs, outputs=prediction)
        ]

        return _ExampleEvalRecord(
            metrics=sample_metrics,
            metadata=metadata,
            sample_outcome=sample_outcome,
            agent_turns=agent_turns,
            round_metric_rows=round_metric_rows,
        )

    def evaluate_single_example(
        self,
        predict_fn: PredictFnAdapter,
        example: dspy.Example,
        extra_metadata: dict[str, Any] | None,
        example_index: int,
    ) -> _ExampleEvalRecord:
        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled by interrupt request.')
        return self._evaluate_single_example(
            predict_fn=predict_fn,
            example=example,
            extra_metadata=extra_metadata,
            example_index=example_index,
        )

    def _evaluate_deterministic(
        self,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled before deterministic execution started.')
        predict_fn = PredictFnAdapter(program)
        parallel_workers = self._resolve_parallel_workers()
        stream_sink, stream_dispatcher = self._build_stream_dispatcher()
        tasks = self._build_parallel_tasks(devset, extra_metadata)

        @impure_safe
        def _run_with_manager(
            concurrency_manager: ConcurrencyManager,
        ) -> list[_ParallelEvalResult]:
            return self._run_deterministic_tasks(
                tasks=tasks,
                predict_fn=predict_fn,
                extra_metadata=extra_metadata,
                parallel_workers=parallel_workers,
                stream_dispatcher=stream_dispatcher,
                concurrency_manager=concurrency_manager,
            )
        
        parallel_results: list[_ParallelEvalResult] = unsafe_perform_io(
            managed(
                _run_with_manager,
                lambda mgr, _: impure_safe(mgr.shutdown)(),
            )(impure_safe(self._build_concurrency_manager)(parallel_workers))
            .alt(raise_exception)
            .unwrap()
        )

        records, failed_examples, sample_outcomes, agent_turns = self._reduce_parallel_results(
            parallel_results
        )
        stream_dispatcher.emit(
            EvaluationCompletedStreamEvent(
                processed_count=len(records),
                failure_count=len(failed_examples),
            )
        )
        stream_dispatcher.close()

        if not records:
            raise ValueError(
                'Deterministic evaluation produced no successful examples; '
                f'failed={len(failed_examples)}.'
            )

        overall_metrics = mean_metric_dict([record.metrics for record in records])
        stratified_rows = self._stratify_records(records)
        output = _EvaluatorOutput(
            backend=EvaluatorBackend.DETERMINISTIC.value,
            system_robustness=self._extract_system_robustness(overall_metrics),
            genai_evaluation=None,
            detailed_results=[
                {
                    'metrics': record.metrics,
                    'metadata': record.metadata.model_dump(mode='python'),
                }
                for record in records
            ],
            config=getattr(program, 'config', None),
            genai_metrics=overall_metrics,
            stratified_metrics=[row.model_dump(mode='python') for row in stratified_rows],
            uncertainty=metric_uncertainty([record.metrics for record in records]),
            failure_count=len(failed_examples),
            processed_count=len(records),
            failed_examples=failed_examples,
            sample_outcomes=sample_outcomes,
            agent_turns=agent_turns,
            analysis_schema=self._analysis_schema(),
            stream_metric_rows=stream_sink.metric_rows,
            stream_round_metric_rows=stream_sink.round_metric_rows,
            stream_failure_rows=stream_sink.failure_rows,
            stream_summary=stream_sink.summary,
        )
        return output.model_dump(mode='python')

    def _build_stream_dispatcher(self) -> tuple[InMemoryMetricEventSink, MetricStreamDispatcher]:
        stream_sink = InMemoryMetricEventSink()
        sinks: list[Any] = [stream_sink]
        if self.local_stream_config is not None:
            sinks.append(JsonlFileMetricEventSink(config=self.local_stream_config))
            if self.local_stream_config.write_csv:
                sinks.append(CsvFileMetricEventSink(config=self.local_stream_config))
        return stream_sink, MetricStreamDispatcher(sinks=sinks, config=self.local_stream_config)

    def _run_deterministic_tasks(
        self,
        *,
        tasks: list[_ParallelEvalTask],
        predict_fn: PredictFnAdapter,
        extra_metadata: dict[str, Any] | None,
        parallel_workers: int,
        stream_dispatcher: MetricStreamDispatcher,
        concurrency_manager: ConcurrencyManager,
    ) -> list[_ParallelEvalResult]:
        if self.deterministic_execution_mode == DeterministicExecutionMode.DSPY_BATCH:
            return self._run_dspy_batch_tasks(
                tasks=tasks,
                predict_fn=predict_fn,
                extra_metadata=extra_metadata,
                parallel_workers=parallel_workers,
                stream_dispatcher=stream_dispatcher,
                concurrency_manager=concurrency_manager,
            )

        return asyncio.run(
            self._run_sliding_window_tasks(
                tasks=tasks,
                predict_fn=predict_fn,
                extra_metadata=extra_metadata,
                parallel_workers=parallel_workers,
                stream_dispatcher=stream_dispatcher,
                concurrency_manager=concurrency_manager,
            )
        )

    def _run_dspy_batch_tasks(
        self,
        *,
        tasks: list[_ParallelEvalTask],
        predict_fn: PredictFnAdapter,
        extra_metadata: dict[str, Any] | None,
        parallel_workers: int,
        stream_dispatcher: MetricStreamDispatcher,
        concurrency_manager: ConcurrencyManager,
    ) -> list[_ParallelEvalResult]:
        batch_workers = min(parallel_workers, concurrency_manager.max_evaluation_threads)
        if batch_workers < parallel_workers:
            logger.warning(
                'Capping dspy batch workers by evaluator bulkhead '
                f'from {parallel_workers} to {batch_workers}.'
            )
        parallel = dspy.Parallel(
            num_threads=batch_workers,
            max_errors=self.parallel_max_errors,
            disable_progress_bar=self.parallel_disable_progress_bar,
        )
        worker = _DeterministicParallelWorker(
            predict_fn=predict_fn,
            evaluator=self,
            extra_metadata=extra_metadata,
        )
        parallel_results = cast(
            list[_ParallelEvalResult],
            parallel([(worker, {'task': task}) for task in tasks]),
        )
        for result in parallel_results:
            self._emit_stream_events_for_result(result=result, stream_dispatcher=stream_dispatcher)
        return parallel_results

    async def _run_sliding_window_tasks(
        self,
        *,
        tasks: list[_ParallelEvalTask],
        predict_fn: PredictFnAdapter,
        extra_metadata: dict[str, Any] | None,
        parallel_workers: int,
        stream_dispatcher: MetricStreamDispatcher,
        concurrency_manager: ConcurrencyManager,
    ) -> list[_ParallelEvalResult]:
        window_size = self._resolve_deterministic_window_size(
            parallel_workers,
            max_evaluation_threads=concurrency_manager.max_evaluation_threads,
        )
        total_tasks = len(tasks)
        logger.info(
            'Starting deterministic sliding-window evaluation '
            f'(window_size={window_size}, task_timeout_s={self.deterministic_task_timeout_seconds:.1f}, '
            f'total_tasks={total_tasks}).'
        )

        async def _evaluate_task(
            task: _ParallelEvalTask,
            *_unused: _ParallelEvalTask,
        ) -> _ParallelEvalResult:
            return await self._evaluate_sliding_task(
                task,
                predict_fn=predict_fn,
                extra_metadata=extra_metadata,
                concurrency_manager=concurrency_manager,
            )

        completed = 0

        async def _heartbeat(stop: asyncio.Event) -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._SLIDING_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    logger.info(
                        'Sliding-window evaluation is still running '
                        f'({completed}/{total_tasks} completed, '
                        f'no completion in {self._SLIDING_HEARTBEAT_SECONDS:.0f}s, '
                        f'python_threads={self._thread_name_snapshot(top_k=self._THREAD_SNAPSHOT_TOP_K)}).'
                    )

        results: list[_ParallelEvalResult] = []
        stop_event = asyncio.Event()
        heartbeat_task = asyncio.create_task(_heartbeat(stop_event))
        mapped: Any = stream.map(
            stream.iterate(tasks), _evaluate_task, ordered=False, task_limit=window_size
        )
        try:
            with tqdm(
                total=total_tasks,
                disable=self.parallel_disable_progress_bar,
                desc='Processed samples',
                unit='sample',
            ) as progress_bar:
                async with mapped.stream() as streamer:
                    async for result in streamer:
                        if self.is_cancel_requested():
                            raise KeyboardInterrupt(
                                'Evaluation cancelled during sliding-window execution.'
                            )
                        completed += 1
                        results.append(result)
                        self._handle_sliding_result(
                            sample_result=result,
                            completed_count=completed,
                            total_tasks=total_tasks,
                            window_size=window_size,
                            stream_dispatcher=stream_dispatcher,
                            progress_bar=progress_bar,
                        )
        finally:
            stop_event.set()
            heartbeat_task.cancel()
            with suppress(Exception):
                await heartbeat_task

        return results

    async def _evaluate_sliding_task(
        self,
        task: _ParallelEvalTask,
        *,
        predict_fn: PredictFnAdapter,
        extra_metadata: dict[str, Any] | None,
        concurrency_manager: ConcurrencyManager,
    ) -> _ParallelEvalResult:
        if self.is_cancel_requested():
            return _ParallelEvalResult(
                task=task,
                record=None,
                error='Evaluation cancelled by interrupt request.',
            )

        @future_safe
        async def _bounded_eval() -> _ExampleEvalRecord:
            return await asyncio.wait_for(
                concurrency_manager.run_in_eval_pool(
                    self.evaluate_single_example,
                    predict_fn,
                    task.example,
                    extra_metadata,
                    task.example_index,
                ),
                timeout=self.deterministic_task_timeout_seconds,
            )

        io_result: IOResult[_ExampleEvalRecord, Exception] = await _bounded_eval().awaitable()
        if is_successful(io_result):
            return _ParallelEvalResult(
                task=task, record=unsafe_perform_io(io_result.unwrap()), error=None
            )
        err: Exception = unsafe_perform_io(io_result.failure())
        return _ParallelEvalResult(
            task=task,
            record=None,
            error=(
                f'Evaluation timed out after {self.deterministic_task_timeout_seconds:.1f}s.'
                if isinstance(err, TimeoutError)
                else str(err)
            ),
        )

    def _handle_sliding_result(
        self,
        *,
        sample_result: _ParallelEvalResult,
        completed_count: int,
        total_tasks: int,
        window_size: int,
        stream_dispatcher: MetricStreamDispatcher,
        progress_bar: Any,
    ) -> None:
        progress_bar.update(1)
        self._emit_stream_events_for_result(
            result=sample_result,
            stream_dispatcher=stream_dispatcher,
        )
        self._log_sliding_progress(
            completed_count=completed_count,
            total_tasks=total_tasks,
            window_size=window_size,
        )

    @staticmethod
    def _log_sliding_progress(
        *,
        completed_count: int,
        total_tasks: int,
        window_size: int,
    ) -> None:
        should_log = completed_count % max(1, window_size) == 0 or completed_count == total_tasks
        if should_log:
            logger.info(f'Processed {completed_count}/{total_tasks} samples (sliding-window).')

    def _resolve_deterministic_window_size(
        self,
        parallel_workers: int,
        *,
        max_evaluation_threads: int,
    ) -> int:
        configured_window = (
            parallel_workers if self.deterministic_window_size is None else self.deterministic_window_size
        )
        return max(1, min(configured_window, max_evaluation_threads))

    @staticmethod
    def _thread_name_snapshot(*, top_k: int) -> str:
        threads = threading.enumerate()
        names = [re.sub(r'-\d+$', '', thread.name) for thread in threads]
        name_counts = Counter(names)
        top = ', '.join(f'{name}:{count}' for name, count in name_counts.most_common(max(1, top_k)))
        return f"tot:{len(threads)} [{top or 'none'}]"

    @staticmethod
    def _emit_stream_events_for_result(
        *,
        result: _ParallelEvalResult,
        stream_dispatcher: MetricStreamDispatcher,
    ) -> None:
        match result:
            case _ParallelEvalResult(task=task, record=_ExampleEvalRecord() as record, error=None):
                stream_dispatcher.emit(
                    SampleMetricsStreamEvent(
                        sample_id=task.sample_id,
                        example_index=task.example_index,
                        metadata=task.metadata.model_dump(mode='python'),
                        metrics=record.metrics,
                        turn_count=record.sample_outcome.turn_count,
                        sample_run_id=record.sample_outcome.sample_run_id,
                    )
                )
                for round_row in record.round_metric_rows:
                    stream_dispatcher.emit(
                        SampleRoundMetricsStreamEvent(
                            sample_id=task.sample_id,
                            example_index=task.example_index,
                            metadata=task.metadata.model_dump(mode='python'),
                            turn_index=round_row.turn_index,
                            robustness_rate=round_row.robustness_rate,
                            bias_prevalence=round_row.bias_prevalence,
                            propagation_pr_t=round_row.propagation_pr_t,
                            first_biased_turn=round_row.first_biased_turn,
                            emergence_observed=round_row.emergence_observed,
                            biased_agent_count=round_row.biased_agent_count,
                            biased_agents=round_row.biased_agents,
                            biased_models=round_row.biased_models,
                            agent_bias_flags=round_row.agent_bias_flags,
                        )
                    )
            case _ParallelEvalResult(task=task, record=None, error=error_text) if (
                error_text is not None
            ):
                stream_dispatcher.emit(
                    SampleFailureStreamEvent(
                        sample_id=task.sample_id,
                        example_index=task.example_index,
                        metadata=task.metadata.model_dump(mode='python'),
                        error=error_text,
                    )
                )
            case _:
                raise TypeError(f'Unexpected stream result payload: {result!r}')

    def _build_parallel_tasks(
        self,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> list[_ParallelEvalTask]:
        return [
            _ParallelEvalTask(
                example_index=index,
                example=example,
                inputs=inputs,
                metadata=self._extract_metadata(inputs, extra_metadata),
                sample_id=self._resolve_sample_id(inputs, index),
            )
            for index, example in enumerate(devset)
            for inputs in [example.toDict()]
        ]

    @staticmethod
    def _reduce_parallel_results(
        parallel_results: list[_ParallelEvalResult],
    ) -> tuple[
        list[_ExampleEvalRecord],
        list[_FailureExampleRow],
        list[_SampleOutcomeRow],
        list[_AgentTurnRow],
    ]:
        records: list[_ExampleEvalRecord] = []
        failed_examples: list[_FailureExampleRow] = []
        sample_outcomes: list[_SampleOutcomeRow] = []
        agent_turns: list[_AgentTurnRow] = []
        for result in sorted(parallel_results, key=lambda r: r.task.example_index):
            match result:
                case _ParallelEvalResult(
                    task=task, record=_ExampleEvalRecord() as record, error=None
                ):
                    records.append(record)
                    sample_outcomes.append(record.sample_outcome)
                    agent_turns.extend(record.agent_turns)
                case _ParallelEvalResult(task=task, record=None, error=error_text) if (
                    error_text is not None
                ):
                    failed_examples.append(
                        _FailureExampleRow(
                            index=task.example_index,
                            error=error_text,
                            sample_id=task.sample_id,
                            metadata=task.metadata,
                        )
                    )
                    sample_outcomes.append(MASEvaluator._failure_sample_outcome(task, error_text))
                case _:
                    raise TypeError(f'Unexpected deterministic parallel result: {result!r}')
        return records, failed_examples, sample_outcomes, agent_turns

    @staticmethod
    def _failure_sample_outcome(task: _ParallelEvalTask, error_message: str) -> _SampleOutcomeRow:
        return _SampleOutcomeRow(
            sample_id=task.sample_id,
            example_index=task.example_index,
            mlflow_run_id=task.metadata.run_id,
            dataset_name=task.metadata.dataset_name,
            dataset_source=task.metadata.dataset_source,
            stereoset_type=task.metadata.stereoset_type,
            category=task.metadata.category,
            protocol=task.metadata.protocol,
            llm_models=task.metadata.llm_models,
            model_names=task.metadata.model_names,
            intervention=task.metadata.intervention,
            num_agents=task.metadata.num_agents,
            rounds=task.metadata.rounds,
            split=task.metadata.split,
            seed=task.metadata.seed,
            question_polarity=str(task.inputs.get('question_polarity', 'unknown')),
            context_condition=str(task.inputs.get('context_condition', 'unknown')),
            label=task.inputs.get('label'),
            gold_answer_text='unknown',
            system_robustness=0.0,
            emergence_rate=0.0,
            amplification_rate=0.0,
            propagation_rate=0.0,
            turn_count=0,
            processed_flag=False,
            failure_reason=error_message,
            sample_run_id=None,
            agent_model_map=task.metadata.agent_model_map,
            first_biased_turn_by_agent={},
            final_is_biased_by_agent={},
            final_answers={},
        )

    def _evaluate_genai(
        self,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        eval_data = self._to_mlflow_eval_dataset(devset)
        predict_fn = PredictFnAdapter(program)
        parallel_workers = self._resolve_parallel_workers()
        with self._with_mlflow_genai_max_workers(parallel_workers):
            genai_result = evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=[
                    system_robustness,
                    emergence_rate,
                    amplification_rate,
                    propagation_rate,
                ],
            )

        run_meta = self._extract_metadata({}, extra_metadata)
        numeric_metrics = {key: float(value) for key, value in genai_result.metrics.items()}
        stratified_rows = [
            _StratifiedRow(
                dimensions={
                    field: str(getattr(run_meta, field)) for field in self._STRATIFY_FIELDS
                },
                support=len(devset),
                metrics=numeric_metrics,
                ci95=dict.fromkeys(numeric_metrics, 0.0),
            )
        ]
        self._validate_stratified_dimensions(stratified_rows)

        output = _EvaluatorOutput(
            backend=EvaluatorBackend.GENAI.value,
            system_robustness=self._extract_system_robustness(genai_result.metrics),
            genai_evaluation=genai_result,
            detailed_results=None,
            config=getattr(program, 'config', None),
            genai_metrics=genai_result.metrics,
            stratified_metrics=[row.model_dump(mode='python') for row in stratified_rows],
            uncertainty=dict.fromkeys(numeric_metrics, 0.0),
            failure_count=0,
            processed_count=len(devset),
            failed_examples=[],
            sample_outcomes=[],
            agent_turns=[],
            analysis_schema=self._analysis_schema(),
            stream_metric_rows=[],
            stream_round_metric_rows=[],
            stream_failure_rows=[],
            stream_summary={'processed_count': len(devset), 'failure_count': 0},
        )
        return output.model_dump(mode='python')

    def __call__(
        self,
        program: dspy.Module,
        devset: list[dspy.Example] | None = None,
        backend: EvaluatorBackend | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled before execution started.')
        devset = devset or self.devset
        chosen_backend = EvaluatorBackend(backend or self.backend)

        strategy = EvaluatorStrategyFactory.create(
            backend=chosen_backend.value,
            deterministic_execute=self._evaluate_deterministic,
            genai_execute=self._evaluate_genai,
        )
        return strategy.evaluate(
            program=program,
            devset=devset,
            extra_metadata=metadata,
        )
