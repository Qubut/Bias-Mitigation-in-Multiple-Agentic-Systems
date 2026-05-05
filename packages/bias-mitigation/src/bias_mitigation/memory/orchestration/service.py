"""Mem0 orchestration service with worker bulkhead and state-driven degrade policy."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

from loguru import logger
from returns.result import Failure, Success

from bias_mitigation.memory.mem0_tools import Mem0Tools

from .models import MemoryOrchestrationConfig, RecallResult
from .statechart import MemoryOrchestrationStateChart


class MemoryOrchestrator:
    """Coordinates memory recall/store through bounded workers and statechart policy."""

    def __init__(
        self,
        *,
        memory_tools: Mem0Tools,
        config: MemoryOrchestrationConfig,
    ):
        self.memory_tools = memory_tools
        self.config = config
        worker_threads = max(1, config.worker_threads)
        self._executor = ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix='Mem0_Orch',
        )
        self._statechart = MemoryOrchestrationStateChart(
            failure_trip_threshold=config.failure_trip_threshold,
            recovery_success_threshold=config.recovery_success_threshold,
        )
        self._pending_store_tasks: set[Future[None]] = set()
        self._pending_lock = Lock()

    @property
    def mode(self) -> str:
        return self._statechart.mode

    def _collect_done_store_futures(self) -> None:
        with self._pending_lock:
            done = {future for future in self._pending_store_tasks if future.done()}
            self._pending_store_tasks.difference_update(done)

    def _can_enqueue_store(self) -> bool:
        self._collect_done_store_futures()
        with self._pending_lock:
            return len(self._pending_store_tasks) < max(1, self.config.max_pending_store_tasks)

    def _record_store_future(self, future: Future[None]) -> None:
        with self._pending_lock:
            self._pending_store_tasks.add(future)

    def _should_skip_store(self) -> bool:
        if self._statechart.mode == 'shed':
            return True
        if self._can_enqueue_store():
            return False
        self._statechart.note_pressure()
        return True

    def _store_payload_task(
        self,
        *,
        payload: str,
        user_id: str,
        metadata: dict[str, Any],
    ) -> None:
        result = self.memory_tools.store_memory(
            content=payload,
            user_id=user_id,
            metadata=metadata,
        )
        if isinstance(result, Failure):
            raise TypeError(str(result.failure()))

    def _attach_async_store_callback(self, future: Future[None]) -> None:
        def _on_done(done_future: Future[None]) -> None:
            try:
                done_future.result()
            except Exception:
                self._statechart.note_failure()
            else:
                self._statechart.note_success()

        future.add_done_callback(_on_done)

    def recall(self, *, question: str, user_id: str, memory_scope: str) -> RecallResult:
        if self._statechart.mode == 'shed':
            return RecallResult(text='No previous statements found.', count=0, status='shed')

        def _search() -> Any:
            return self.memory_tools.search_memories(
                query=question,
                user_id=user_id,
                filters={'memory_scope': memory_scope},
            )

        future = self._executor.submit(_search)
        try:
            result = future.result(timeout=max(0.1, self.config.recall_timeout_ms / 1000.0))
        except Exception as error:
            logger.warning(f'[MemoryOrchestrator]: recall failed ({self.mode}): {error}')
            self._statechart.note_failure()
            return RecallResult(text='No previous statements found.', count=0, status='error')

        match result:
            case Success(memories):
                passages = memories.get('passages', [])
                rendered = self.memory_tools.render_recalled_memory_text(passages)
                self._statechart.note_success()
                status = 'retrieved' if passages else 'empty'
                return RecallResult(text=rendered, count=len(passages), status=status)
            case Failure(error):
                logger.warning(f'[MemoryOrchestrator]: recall backend failure ({self.mode}): {error}')
                self._statechart.note_failure()
                return RecallResult(text='No previous statements found.', count=0, status='error')
            case _:
                self._statechart.note_failure()
                return RecallResult(text='No previous statements found.', count=0, status='unknown')

    def store(
        self,
        *,
        question: str,
        answer: str,
        reasoning: str,
        user_id: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._should_skip_store():
            return

        payload = self.memory_tools.format_store_memory_text(
            question=question,
            answer=answer,
            reasoning=reasoning,
        )

        future = self._executor.submit(
            self._store_payload_task,
            payload=payload,
            user_id=user_id,
            metadata=metadata,
        )
        self._record_store_future(future)

        if not self.config.store_async:
            try:
                future.result(timeout=max(0.1, self.config.store_timeout_ms / 1000.0))
                self._statechart.note_success()
            except Exception:
                self._statechart.note_failure()
            return

        self._attach_async_store_callback(future)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
