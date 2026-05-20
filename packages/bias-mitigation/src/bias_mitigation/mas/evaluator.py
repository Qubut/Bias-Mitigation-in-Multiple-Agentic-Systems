"""MAS evaluation. Runs dspy.Evaluate, streams per-sample rows, stratifies."""

import os
import threading
from collections.abc import Callable
from typing import Any, ClassVar, cast

import dspy
from returns.result import Success, safe

from .evaluation.adapters import PredictFnAdapter
from .evaluation.aggregation import (
    aggregate_stratified_metric_rows,
    mean_metric_dict,
    metric_uncertainty,
    validate_stratified_dimensions,
)
from .evaluation.metadata import MetadataExtractor
from .evaluation.models import (
    _EvaluatorOutput,
    _FailureExampleRow,
    _SampleOutcomeRow,
    _StratifiedRow,
)
from .evaluation.pipeline import (
    build_parallel_tasks,
    build_stream_dispatcher,
    emit_stream_events_for_result,
    record_failure,
)
from .evaluation.scoring import Scorer
from .evaluation.streaming import EvaluationCompletedStreamEvent, LocalStreamConfig
from .evaluation.worker import _ExampleEvalRecord, _ParallelEvalResult


def _safe_program(program: dspy.Module) -> Callable[..., Any]:
    # dspy.Evaluate's ParallelExecutor catches exceptions and hands the metric
    # an empty Prediction, losing the cause. Stuff the error into the Prediction
    # instead so the metric callback can route it through stream_failure_rows.
    wrapped: Any = safe(program)

    def _call(**kwargs: Any) -> Any:
        return (
            wrapped(**kwargs)
            .lash(lambda exc: Success(dspy.Prediction(_eval_error=str(exc))))
            .unwrap()
        )

    return _call


class MASEvaluator:
    """Score a DSPy MAS program over a devset.

    Cancellation is cooperative: another process sets
    BIAS_MITIGATION_CANCEL_REQUESTED=1 and the next checkpoint bails out.
    """

    _CANCEL_ENV_VAR: ClassVar[str] = 'BIAS_MITIGATION_CANCEL_REQUESTED'
    _DEFAULT_MAX_WORKERS_CAP: ClassVar[int] = 8

    def __init__(
        self,
        devset: list[dspy.Example],
        run_metadata: dict[str, Any] | None = None,
        parallel_num_threads: int | None = None,
        parallel_max_errors: int | None = None,
        parallel_disable_progress_bar: bool = False,
        local_stream_config: LocalStreamConfig | None = None,
        index_offset: int = 0,
    ):
        self.devset = devset
        self.run_metadata = run_metadata or {}
        self.parallel_num_threads = parallel_num_threads
        self.parallel_max_errors = parallel_max_errors
        self.parallel_disable_progress_bar = parallel_disable_progress_bar
        self.index_offset = index_offset
        self.local_stream_config = local_stream_config
        self.metadata_extractor = MetadataExtractor(self.run_metadata)
        self.scorer = Scorer(self.metadata_extractor)

    @classmethod
    def _default_parallel_workers(cls) -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, cls._DEFAULT_MAX_WORKERS_CAP))

    def _resolve_parallel_workers(self) -> int:
        if self.parallel_num_threads is not None:
            return self.parallel_num_threads
        return self._default_parallel_workers()

    def is_cancel_requested(self) -> bool:
        """True if BIAS_MITIGATION_CANCEL_REQUESTED=1."""
        return os.getenv(self._CANCEL_ENV_VAR, '0') == '1'

    def _strata_key(self, record: _ExampleEvalRecord) -> tuple[str, ...]:
        return tuple(
            str(getattr(record.metadata, field))
            for field in self.metadata_extractor.STRATIFY_FIELDS
        )

    def _validate_stratified_dimensions(self, rows: list[_StratifiedRow]) -> None:
        validate_stratified_dimensions(
            rows=[{'dimensions': row.dimensions} for row in rows],
            expected_fields=self.metadata_extractor.STRATIFY_FIELDS,
        )

    def _stratify_records(
        self,
        records: list[_ExampleEvalRecord],
    ) -> list[_StratifiedRow]:
        aggregate_rows = aggregate_stratified_metric_rows(
            records=[
                {
                    'metadata': {
                        **record.metadata.model_dump(mode='python'),
                        'y_true': record.y_true,
                        'y_pred': record.y_pred,
                    },
                    'metrics': record.metrics,
                }
                for record in records
            ],
            stratify_fields=self.metadata_extractor.STRATIFY_FIELDS,
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

    def evaluate_single_example(
        self,
        predict_fn: PredictFnAdapter,
        example: dspy.Example,
        extra_metadata: dict[str, Any] | None,
        example_index: int,
    ) -> _ExampleEvalRecord:
        """Single-example path, for debugging. Honours cancel."""
        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled by interrupt request.')
        return self.scorer.evaluate_one(
            predict_fn=predict_fn,
            example=example,
            extra_metadata=extra_metadata,
            example_index=example_index,
        )

    def _evaluate(
        self,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled before deterministic execution started.')
        parallel_workers = self._resolve_parallel_workers()
        stream_sink, stream_dispatcher = build_stream_dispatcher(self.local_stream_config)
        tasks = build_parallel_tasks(
            devset=devset,
            extra_metadata=extra_metadata,
            metadata_extractor=self.metadata_extractor,
            index_offset=self.index_offset,
        )

        # sample_id has to live on inputs because dspy.Evaluate calls
        # program(**example.inputs()) and there's no other way through.
        examples_for_eval = [
            dspy.Example(**task.inputs, sample_id=task.sample_id).with_inputs(
                *task.inputs.keys(), 'sample_id'
            )
            for task in tasks
        ]
        task_by_example_id = {
            id(example): task for example, task in zip(examples_for_eval, tasks, strict=True)
        }

        records_by_index: dict[int, _ExampleEvalRecord] = {}
        failed_examples: list[_FailureExampleRow] = []
        failed_outcomes: list[_SampleOutcomeRow] = []
        metric_lock = threading.Lock()
        robustness_key = Scorer.METRIC_NAME_MAP['system_robustness']

        def metric(example: dspy.Example, prediction: Any) -> float:
            if self.is_cancel_requested():
                raise KeyboardInterrupt('Evaluation cancelled during deterministic execution.')
            task = task_by_example_id[id(example)]
            eval_error = getattr(prediction, '_eval_error', None)
            if eval_error is not None:
                record_failure(
                    task=task,
                    error=str(eval_error),
                    stream_dispatcher=stream_dispatcher,
                    failed_examples=failed_examples,
                    failed_outcomes=failed_outcomes,
                    lock=metric_lock,
                )
                return 0.0
            try:
                record = self.scorer.score_prediction(
                    inputs=task.inputs,
                    prediction=prediction,
                    sample_id=task.sample_id,
                    example_index=task.example_index,
                    extra_metadata=extra_metadata,
                )
            except Exception as exc:
                record_failure(
                    task=task,
                    error=str(exc),
                    stream_dispatcher=stream_dispatcher,
                    failed_examples=failed_examples,
                    failed_outcomes=failed_outcomes,
                    lock=metric_lock,
                )
                return 0.0
            with metric_lock:
                records_by_index[task.example_index] = record
            emit_stream_events_for_result(
                result=_ParallelEvalResult(task=task, record=record, error=None),
                stream_dispatcher=stream_dispatcher,
            )
            return float(record.metrics.get(robustness_key, 0.0))

        dspy.Evaluate(
            devset=examples_for_eval,
            metric=metric,
            num_threads=parallel_workers,
            display_progress=not self.parallel_disable_progress_bar,
            max_errors=self.parallel_max_errors,
            provide_traceback=True,
        )(_safe_program(program))

        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled during deterministic execution.')

        records = [records_by_index[i] for i in sorted(records_by_index)]
        success_outcomes = [record.sample_outcome for record in records]
        sample_outcomes = sorted(
            success_outcomes + failed_outcomes, key=lambda row: row.example_index
        )
        agent_turns = [turn for record in records for turn in record.agent_turns]

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
            system_robustness=float(
                overall_metrics.get(Scorer.METRIC_NAME_MAP['system_robustness'], 0.0)
            ),
            detailed_results=[
                {
                    'metrics': record.metrics,
                    'metadata': record.metadata.model_dump(mode='python'),
                }
                for record in records
            ],
            config=getattr(program, 'config', None),
            overall_metrics=overall_metrics,
            stratified_metrics=[row.model_dump(mode='python') for row in stratified_rows],
            uncertainty=metric_uncertainty([record.metrics for record in records]),
            failure_count=len(failed_examples),
            processed_count=len(records),
            failed_examples=failed_examples,
            sample_outcomes=sample_outcomes,
            agent_turns=agent_turns,
            analysis_schema=self.metadata_extractor.analysis_schema(),
            stream_metric_rows=stream_sink.metric_rows,
            stream_round_metric_rows=stream_sink.round_metric_rows,
            stream_failure_rows=stream_sink.failure_rows,
            stream_summary=stream_sink.summary,
        )
        return output.model_dump(mode='python')

    def __call__(
        self,
        program: dspy.Module,
        devset: list[dspy.Example] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One evaluation pass. devset/metadata override the construction defaults."""
        if self.is_cancel_requested():
            raise KeyboardInterrupt('Evaluation cancelled before execution started.')
        return self._evaluate(
            program=program,
            devset=devset or self.devset,
            extra_metadata=metadata,
        )
