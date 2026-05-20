"""Task construction, stream dispatch, and failure handling for the evaluator.

These helpers sit between the scoring layer (which is pure) and the
streaming sinks (which write to disk / memory).  They turn raw devsets
into the pre-computed task envelopes the metric callback expects, fan
per-sample success/failure into stream events, and capture failures
into the run-level lists that the evaluator returns.

Splitting these out of :class:`MASEvaluator` keeps the evaluator focused
on orchestration (run config, lifecycle, output assembly) while making
the per-sample side-effect surface trivially unit-testable.
"""

from __future__ import annotations

import threading
from typing import Any

import dspy

from .metadata import MetadataExtractor
from .models import _FailureExampleRow, _SampleOutcomeRow
from .streaming import (
    CsvFileMetricEventSink,
    InMemoryMetricEventSink,
    JsonlFileMetricEventSink,
    LocalStreamConfig,
    MetricStreamDispatcher,
    SampleFailureStreamEvent,
    SampleMetricsStreamEvent,
    SampleRoundMetricsStreamEvent,
)
from .worker import _ExampleEvalRecord, _ParallelEvalResult, _ParallelEvalTask


def build_stream_dispatcher(
    local_stream_config: LocalStreamConfig | None,
) -> tuple[InMemoryMetricEventSink, MetricStreamDispatcher]:
    """Construct the metric stream dispatcher for one evaluation run.

    Always installs an in-memory sink so the final evaluator output can
    carry stream rows back to the caller in process.  When
    ``local_stream_config`` is set, a JSONL sink is appended for durable
    on-disk capture and — when the config opts in via ``write_csv`` — a
    CSV mirror is added for ad-hoc analysis.

    Args:
        local_stream_config: Stream-persistence config from the
            evaluator.  ``None`` keeps streaming purely in-memory.

    Returns:
        ``(in_memory_sink, dispatcher)``.  The sink is exposed
        separately so the caller can attach its accumulated rows to the
        final evaluator output once the run completes.
    """
    stream_sink = InMemoryMetricEventSink()
    sinks: list[Any] = [stream_sink]
    if local_stream_config is not None:
        sinks.append(JsonlFileMetricEventSink(config=local_stream_config))
        if local_stream_config.write_csv:
            sinks.append(CsvFileMetricEventSink(config=local_stream_config))
    return stream_sink, MetricStreamDispatcher(sinks=sinks, config=local_stream_config)


def build_parallel_tasks(
    *,
    devset: list[dspy.Example],
    extra_metadata: dict[str, Any] | None,
    metadata_extractor: MetadataExtractor,
    index_offset: int = 0,
) -> list[_ParallelEvalTask]:
    """Materialise the per-example tasks fed to ``dspy.Evaluate``.

    Pre-computes the stable sample id and full evaluation metadata for
    every example so the parallel metric callback can recover them via
    cheap ``id()`` lookup rather than re-extracting under contention.
    The same task envelope is also handed back to stream sinks and to
    the failure recorder, so the ids are consistent across every
    artefact produced by the run.

    Args:
        devset: Examples to evaluate.
        extra_metadata: Optional per-call metadata overrides merged on
            top of the run-wide defaults inside the extractor.
        metadata_extractor: Shared :class:`MetadataExtractor` instance.
        index_offset: Offset applied to each example's local index so
            shards of a larger dataset keep unique global indices.

    Returns:
        One :class:`_ParallelEvalTask` per example, in devset order.
    """
    return [
        _ParallelEvalTask(
            example_index=index_offset + index,
            example=example,
            inputs=inputs,
            metadata=metadata_extractor.extract(inputs, extra_metadata),
            sample_id=metadata_extractor.resolve_sample_id(inputs, index_offset + index),
        )
        for index, example in enumerate(devset)
        for inputs in [example.toDict()]
    ]


def emit_stream_events_for_result(
    *,
    result: _ParallelEvalResult,
    stream_dispatcher: MetricStreamDispatcher,
) -> None:
    """Fan one parallel result out to the metric stream sinks.

    Success results emit a per-sample metrics event plus one per-round
    event so dashboards can observe both aggregate sample scores and
    turn-level bias propagation.  Failure results emit a single failure
    event with the captured error text.  Any unexpected payload shape
    is treated as a programmer error and raised so it cannot silently
    corrupt downstream tables.

    Args:
        result: Outcome of a single parallel evaluation task.
        stream_dispatcher: Active dispatcher fanning events out to every
            configured sink.

    Raises:
        TypeError: If ``result`` is neither a success nor a failure,
            indicating a bug upstream.
    """
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


def failure_sample_outcome(task: _ParallelEvalTask, error_message: str) -> _SampleOutcomeRow:
    """Build a placeholder sample outcome row for a failed task.

    Keeping failed samples in the outcomes table (with metric values
    zeroed and ``processed_flag=False``) preserves a single source of
    truth for "which samples were attempted" and lets downstream tools
    detect partial coverage in fairness reports.

    Args:
        task: The task that failed.
        error_message: Human-readable failure reason.

    Returns:
        A :class:`_SampleOutcomeRow` that mirrors the success schema but
        carries empty metrics and the failure reason.
    """
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


def record_failure(
    *,
    task: _ParallelEvalTask,
    error: str,
    stream_dispatcher: MetricStreamDispatcher,
    failed_examples: list[_FailureExampleRow],
    failed_outcomes: list[_SampleOutcomeRow],
    lock: threading.Lock,
) -> None:
    """Append a failure row pair under ``lock`` and emit a stream event.

    Called from the ``dspy.Evaluate`` metric callback on worker threads,
    so the failure-list appends are guarded by the shared lock.  The
    stream event itself is emitted outside the lock because
    :class:`MetricStreamDispatcher` is already thread-safe and we don't
    want to hold the metric lock across an I/O-bound dispatch.

    Args:
        task: The task that failed.
        error: Human-readable failure reason.
        stream_dispatcher: Active dispatcher fanning events to sinks.
        failed_examples: Accumulator the evaluator returns to its caller.
        failed_outcomes: Outcome-row accumulator (parallel to
            ``failed_examples``).
        lock: Mutex protecting the two accumulators.
    """
    with lock:
        failed_examples.append(
            _FailureExampleRow(
                index=task.example_index,
                error=error,
                sample_id=task.sample_id,
                metadata=task.metadata,
            )
        )
        failed_outcomes.append(failure_sample_outcome(task, error))
    emit_stream_events_for_result(
        result=_ParallelEvalResult(task=task, record=None, error=error),
        stream_dispatcher=stream_dispatcher,
    )
