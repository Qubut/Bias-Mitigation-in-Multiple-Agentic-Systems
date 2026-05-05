"""Streaming contracts and sinks for per-sample evaluation telemetry."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Full, Queue
from threading import Lock
from typing import Any, Protocol

from aiostream import stream


@dataclass(frozen=True, slots=True)
class SampleMetricsStreamEvent:
    """Successful per-sample metric event emitted during evaluation."""

    sample_id: str
    example_index: int
    metadata: dict[str, str]
    metrics: dict[str, float]
    turn_count: int
    sample_run_id: str | None


@dataclass(frozen=True, slots=True)
class SampleFailureStreamEvent:
    """Failure event emitted when one sample evaluation fails."""

    sample_id: str
    example_index: int
    metadata: dict[str, str]
    error: str


@dataclass(frozen=True, slots=True)
class SampleRoundMetricsStreamEvent:
    """Per-turn metric event emitted for one sample/turn pair."""

    sample_id: str
    example_index: int
    metadata: dict[str, str]
    turn_index: int
    robustness_rate: float
    bias_prevalence: float
    propagation_pr_t: float
    first_biased_turn: int
    emergence_observed: bool
    biased_agent_count: int
    biased_agents: list[str]
    biased_models: list[str]
    agent_bias_flags: dict[str, bool]


@dataclass(frozen=True, slots=True)
class EvaluationCompletedStreamEvent:
    """Terminal event emitted when the deterministic loop finishes."""

    processed_count: int
    failure_count: int


EvalStreamEvent = (
    SampleMetricsStreamEvent
    | SampleFailureStreamEvent
    | SampleRoundMetricsStreamEvent
    | EvaluationCompletedStreamEvent
)


class MetricEventSink(Protocol):
    """Sink protocol for declarative stream event handling."""

    async def handle(self, event: EvalStreamEvent) -> None:
        """Consume one stream event."""

    async def flush(self) -> None:
        """Flush buffered writes after stream completion."""


@dataclass(frozen=True, slots=True)
class LocalStreamConfig:
    """Configuration for durable local stream persistence."""

    root_dir: str
    run_id: str
    flush_every_events: int = 1
    fsync: bool = False
    write_csv: bool = True
    run_dir_name: str | None = None
    run_manifest: dict[str, Any] | None = None
    write_manifest: bool = True
    index_filename: str = 'runs_index.jsonl'
    max_buffered_events: int = 2048
    drop_events_on_backpressure: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.root_dir) / (self.run_dir_name or self.run_id)

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / 'run_manifest.json'

    @property
    def root_index_path(self) -> Path:
        return Path(self.root_dir) / self.index_filename

    @property
    def metric_jsonl_path(self) -> Path:
        return self.run_dir / 'stream_metric_rows.jsonl'

    @property
    def failure_jsonl_path(self) -> Path:
        return self.run_dir / 'stream_failure_rows.jsonl'

    @property
    def round_metric_jsonl_path(self) -> Path:
        return self.run_dir / 'stream_round_metrics.jsonl'

    @property
    def metric_csv_path(self) -> Path:
        return self.run_dir / 'stream_metric_rows.csv'

    @property
    def failure_csv_path(self) -> Path:
        return self.run_dir / 'stream_failure_rows.csv'

    @property
    def round_metric_csv_path(self) -> Path:
        return self.run_dir / 'stream_round_metrics.csv'

    @property
    def summary_path(self) -> Path:
        return self.run_dir / 'stream_summary.json'


@dataclass(frozen=True, slots=True)
class _MetricRow:
    """Normalized row envelope for successful stream events."""

    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _FailureRow:
    """Normalized row envelope for failure stream events."""

    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RoundMetricRow:
    """Normalized row envelope for round-level metric stream events."""

    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SummaryRow:
    """Normalized row envelope for completion summary."""

    row: dict[str, int]


def _slugify_token(value: Any, max_length: int) -> str:
    raw = str(value or '').strip().lower()
    normalized = re.sub(r'[^a-z0-9_.-]+', '-', raw).strip('-_.')
    if not normalized:
        return 'na'
    return normalized[:max_length]


def build_live_run_dir_name(
    *,
    template: str,
    tokens: dict[str, Any],
    token_max_length: int,
) -> str:
    """Build deterministic, readable live directory name from runtime tokens."""
    slugged_tokens = {
        key: _slugify_token(value, token_max_length)
        for key, value in tokens.items()
    }
    try:
        rendered = template.format(**slugged_tokens)
    except KeyError:
        rendered = '{started_at}_{run_name}_{intervention}_{run_id_short}'.format(**slugged_tokens)

    normalized_rendered = re.sub(r'[^a-z0-9_.-]+', '-', rendered.lower()).strip('-_.')
    return normalized_rendered or slugged_tokens.get('run_id_short', slugged_tokens['run_id'])


def initialize_local_stream_layout(config: LocalStreamConfig) -> None:
    """Create live stream directory and traceability artifacts for one run."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        'run_id': config.run_id,
        'run_dir_name': config.run_dir.name,
        'run_dir_path': str(config.run_dir),
        **(config.run_manifest or {}),
    }

    if config.write_manifest:
        config.run_manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    with config.root_index_path.open('a', encoding='utf-8') as index_file:
        index_file.write(json.dumps(manifest_payload, ensure_ascii=False) + '\n')


def _normalize_event_row(
    event: EvalStreamEvent,
    emitted_at: str,
) -> _MetricRow | _FailureRow | _RoundMetricRow | _SummaryRow:
    """Normalize stream events into typed row payloads using pattern matching."""
    match event:
        case SampleMetricsStreamEvent(
            sample_id=sample_id,
            example_index=example_index,
            metadata=metadata,
            metrics=metrics,
            turn_count=turn_count,
            sample_run_id=sample_run_id,
        ):
            return _MetricRow(
                row={
                    'sample_id': sample_id,
                    'example_index': example_index,
                    'sample_run_id': sample_run_id,
                    'turn_count': turn_count,
                    'emitted_at_utc': emitted_at,
                    **metadata,
                    **metrics,
                }
            )
        case SampleFailureStreamEvent(
            sample_id=sample_id,
            example_index=example_index,
            metadata=metadata,
            error=error,
        ):
            return _FailureRow(
                row={
                    'sample_id': sample_id,
                    'example_index': example_index,
                    'error': error,
                    'emitted_at_utc': emitted_at,
                    **metadata,
                }
            )
        case SampleRoundMetricsStreamEvent(
            sample_id=sample_id,
            example_index=example_index,
            metadata=metadata,
            turn_index=turn_index,
            robustness_rate=robustness_rate,
            bias_prevalence=bias_prevalence,
            propagation_pr_t=propagation_pr_t,
            first_biased_turn=first_biased_turn,
            emergence_observed=emergence_observed,
            biased_agent_count=biased_agent_count,
            biased_agents=biased_agents,
            biased_models=biased_models,
            agent_bias_flags=agent_bias_flags,
        ):
            return _RoundMetricRow(
                row={
                    'sample_id': sample_id,
                    'example_index': example_index,
                    'turn_index': turn_index,
                    'robustness_rate': robustness_rate,
                    'bias_prevalence': bias_prevalence,
                    'propagation_pr_t': propagation_pr_t,
                    'first_biased_turn': first_biased_turn,
                    'emergence_observed': emergence_observed,
                    'biased_agent_count': biased_agent_count,
                    'biased_agents': biased_agents,
                    'biased_models': biased_models,
                    'agent_bias_flags': agent_bias_flags,
                    'emitted_at_utc': emitted_at,
                    **metadata,
                }
            )
        case EvaluationCompletedStreamEvent(
            processed_count=processed_count,
            failure_count=failure_count,
        ):
            return _SummaryRow(
                row={
                    'processed_count': processed_count,
                    'failure_count': failure_count,
                }
            )


@dataclass(slots=True)
class InMemoryMetricEventSink:
    """Simple sink capturing stream events for later artifact persistence."""

    metric_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    round_metric_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=lambda: {'processed_count': 0, 'failure_count': 0})

    async def handle(self, event: EvalStreamEvent) -> None:
        """Consume one event and store normalized sink rows."""
        emitted_at = datetime.now(tz=UTC).isoformat()
        normalized = _normalize_event_row(event, emitted_at)
        match normalized:
            case _MetricRow(row=row):
                self.metric_rows.append(row)
            case _FailureRow(row=row):
                self.failure_rows.append(row)
            case _RoundMetricRow(row=row):
                self.round_metric_rows.append(row)
            case _SummaryRow(row=row):
                self.summary = row

    async def flush(self) -> None:
        """No-op for in-memory sink."""


@dataclass(slots=True)
class JsonlFileMetricEventSink:
    """Thread-safe durable sink writing live stream events to local JSONL files."""

    config: LocalStreamConfig
    _metric_file: Any = field(init=False, default=None)
    _failure_file: Any = field(init=False, default=None)
    _round_metric_file: Any = field(init=False, default=None)
    _events_since_flush: int = field(init=False, default=0)
    _lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self._metric_file = self.config.metric_jsonl_path.open('a', encoding='utf-8')
        self._failure_file = self.config.failure_jsonl_path.open('a', encoding='utf-8')
        self._round_metric_file = self.config.round_metric_jsonl_path.open('a', encoding='utf-8')

    @staticmethod
    def _write_line(file_handle: Any, payload: dict[str, Any]) -> None:
        file_handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def _flush_locked(self) -> None:
        self._metric_file.flush()
        self._failure_file.flush()
        self._round_metric_file.flush()
        if self.config.fsync:
            os.fsync(self._metric_file.fileno())
            os.fsync(self._failure_file.fileno())
            os.fsync(self._round_metric_file.fileno())
        self._events_since_flush = 0

    async def handle(self, event: EvalStreamEvent) -> None:
        emitted_at = datetime.now(tz=UTC).isoformat()
        normalized = _normalize_event_row(event, emitted_at)

        with self._lock:
            match normalized:
                case _MetricRow(row=row):
                    self._write_line(self._metric_file, row)
                    self._events_since_flush += 1
                case _FailureRow(row=row):
                    self._write_line(self._failure_file, row)
                    self._events_since_flush += 1
                case _RoundMetricRow(row=row):
                    self._write_line(self._round_metric_file, row)
                    self._events_since_flush += 1
                case _SummaryRow(row=row):
                    self.config.summary_path.write_text(
                        json.dumps(row, ensure_ascii=False, indent=2),
                        encoding='utf-8',
                    )

            if self._events_since_flush >= max(1, self.config.flush_every_events):
                self._flush_locked()

    async def flush(self) -> None:
        with self._lock:
            self._flush_locked()


@dataclass(slots=True)
class CsvFileMetricEventSink:
    """Thread-safe sink writing live stream events to local CSV files."""

    config: LocalStreamConfig
    _metric_file: Any = field(init=False, default=None)
    _failure_file: Any = field(init=False, default=None)
    _round_metric_file: Any = field(init=False, default=None)
    _metric_writer: Any = field(init=False, default=None)
    _failure_writer: Any = field(init=False, default=None)
    _round_metric_writer: Any = field(init=False, default=None)
    _metric_header: list[str] | None = field(init=False, default=None)
    _failure_header: list[str] | None = field(init=False, default=None)
    _round_metric_header: list[str] | None = field(init=False, default=None)
    _events_since_flush: int = field(init=False, default=0)
    _lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self._metric_file = self.config.metric_csv_path.open('a', newline='', encoding='utf-8')
        self._failure_file = self.config.failure_csv_path.open('a', newline='', encoding='utf-8')
        self._round_metric_file = self.config.round_metric_csv_path.open('a', newline='', encoding='utf-8')
        self._metric_writer = csv.writer(self._metric_file)
        self._failure_writer = csv.writer(self._failure_file)
        self._round_metric_writer = csv.writer(self._round_metric_file)

    @staticmethod
    def _write_row(
        writer: Any,
        header: list[str] | None,
        file_handle: Any,
        row: dict[str, Any],
    ) -> tuple[Any, list[str]]:
        current_header = header
        if current_header is None:
            current_header = list(row.keys())
            writer.writerow(current_header)
        writer.writerow([row.get(column) for column in current_header])
        return writer, current_header

    def _flush_locked(self) -> None:
        self._metric_file.flush()
        self._failure_file.flush()
        self._round_metric_file.flush()
        if self.config.fsync:
            os.fsync(self._metric_file.fileno())
            os.fsync(self._failure_file.fileno())
            os.fsync(self._round_metric_file.fileno())
        self._events_since_flush = 0

    async def handle(self, event: EvalStreamEvent) -> None:
        emitted_at = datetime.now(tz=UTC).isoformat()
        normalized = _normalize_event_row(event, emitted_at)

        with self._lock:
            match normalized:
                case _MetricRow(row=row):
                    self._metric_writer, self._metric_header = self._write_row(
                        self._metric_writer,
                        self._metric_header,
                        self._metric_file,
                        row,
                    )
                    self._events_since_flush += 1
                case _FailureRow(row=row):
                    self._failure_writer, self._failure_header = self._write_row(
                        self._failure_writer,
                        self._failure_header,
                        self._failure_file,
                        row,
                    )
                    self._events_since_flush += 1
                case _RoundMetricRow(row=row):
                    self._round_metric_writer, self._round_metric_header = self._write_row(
                        self._round_metric_writer,
                        self._round_metric_header,
                        self._round_metric_file,
                        row,
                    )
                    self._events_since_flush += 1
                case _SummaryRow():
                    pass

            if self._events_since_flush >= max(1, self.config.flush_every_events):
                self._flush_locked()

    async def flush(self) -> None:
        with self._lock:
            self._flush_locked()


@dataclass(slots=True)
class MetricStreamDispatcher:
    """Synchronous facade dispatching events through aiostream pipeline."""

    sinks: Sequence[MetricEventSink]
    config: LocalStreamConfig | None = None
    _background_tasks: set[asyncio.Task[Any]] = field(init=False, default_factory=set)
    _event_queue: Queue[EvalStreamEvent | None] = field(init=False)
    _dropped_events: int = field(init=False, default=0)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        queue_size = 2048
        if self.config is not None:
            queue_size = max(1, self.config.max_buffered_events)
        self._event_queue = Queue(maxsize=queue_size)

    async def _dispatch_to_sinks(self, event: EvalStreamEvent) -> None:
        await asyncio.gather(*(sink.handle(event) for sink in self.sinks))

    async def _drain_queue_async(self, events: list[EvalStreamEvent]) -> None:
        source = stream.iterate(events)
        async with source.stream() as streamer:
            async for emitted_event in streamer:
                await self._dispatch_to_sinks(self._coerce_stream_event(emitted_event))

    @staticmethod
    def _coerce_stream_event(event: object) -> EvalStreamEvent:
        match event:
            case (
                SampleMetricsStreamEvent()
                | SampleFailureStreamEvent()
                | SampleRoundMetricsStreamEvent()
                | EvaluationCompletedStreamEvent()
            ):
                return event
            case _:
                raise TypeError(f'Unsupported stream event type: {type(event)!r}')

    async def _emit_async(self, event: EvalStreamEvent) -> None:
        source = stream.just(event)
        async with source.stream() as streamer:
            async for emitted_event in streamer:
                await self._dispatch_to_sinks(self._coerce_stream_event(emitted_event))

    async def _flush_async(self) -> None:
        await asyncio.gather(*(sink.flush() for sink in self.sinks))

    def _run_coro(self, coro: Coroutine[Any, Any, Any]) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            _ = asyncio.run(coro)
            return

        task = running_loop.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _enqueue_event(self, event: EvalStreamEvent) -> None:
        if self._closed:
            return
        should_drop = bool(self.config and self.config.drop_events_on_backpressure)
        try:
            if should_drop:
                self._event_queue.put_nowait(event)
            else:
                self._event_queue.put(event)
        except Full:
            self._dropped_events += 1

    def _drain_queue_sync(self) -> None:
        drained: list[EvalStreamEvent] = []
        while not self._event_queue.empty():
            maybe_event = self._event_queue.get()
            if maybe_event is None:
                continue
            drained.append(maybe_event)
        if not drained:
            return
        self._run_coro(self._drain_queue_async(drained))

    def emit(self, event: EvalStreamEvent) -> None:
        """Emit one event to every registered sink."""
        self._enqueue_event(event)
        self._drain_queue_sync()

    def close(self) -> None:
        """Flush all sinks at end of stream."""
        self._closed = True
        self._drain_queue_sync()
        self._run_coro(self._flush_async())
