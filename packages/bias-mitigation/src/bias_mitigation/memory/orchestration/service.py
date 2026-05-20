"""Mem0 orchestration service with cross-loop bulkhead and state-driven degrade policy.

The orchestrator wraps a :class:`Mem0Tools` (which now drives
:class:`mem0.AsyncMemory` natively) with:

* a state-machine that decides whether to ``shed`` calls or attempt
  recovery, fed by recall/store success/failure events,
* a bounded :class:`threading.BoundedSemaphore` to cap concurrent
  in-flight store tasks (the equivalent of the legacy
  ``max_pending_store_tasks`` thread-pool bound).  ``threading`` rather
  than ``asyncio`` because :func:`dspy.syncify` runs each program call
  in a fresh event loop, and ``asyncio.Semaphore`` is loop-local —
  it would either raise ``RuntimeError: bound to a different event
  loop`` or silently time out (tripping the upstream pressure breaker)
  on every call after the first worker thread, and
* per-call ``asyncio.wait_for`` timeouts for recall and store.
"""

from __future__ import annotations

import asyncio
from threading import BoundedSemaphore
from typing import Any

from loguru import logger
from returns.result import Failure, Success

from bias_mitigation.data.models.config import MemoryOrchestrationConfig
from bias_mitigation.memory.mem0_tools import Mem0Tools

from .models import RecallResult
from .statechart import MemoryOrchestrationStateChart


class MemoryOrchestrator:
    """Coordinates async memory recall/store under bounded concurrency and statechart policy."""

    def __init__(
        self,
        *,
        memory_tools: Mem0Tools,
        config: MemoryOrchestrationConfig,
    ):
        self.memory_tools = memory_tools
        self.config = config
        self._statechart = MemoryOrchestrationStateChart(
            failure_trip_threshold=config.failure_trip_threshold,
            recovery_success_threshold=config.recovery_success_threshold,
        )
        # Cross-thread bulkhead — see module docstring for why this can't be
        # an ``asyncio.Semaphore`` under :func:`dspy.syncify`.
        self._store_semaphore = BoundedSemaphore(max(1, config.max_pending_store_tasks))
        # Tracks how many store tasks currently hold the semaphore so
        # ``_should_skip_store`` can shed when the pool is saturated.
        self._store_inflight = 0

    @property
    def mode(self) -> str:
        return self._statechart.mode

    def _should_skip_store(self) -> bool:
        if self._statechart.mode == 'shed':
            return True
        if self._store_inflight >= max(1, self.config.max_pending_store_tasks):
            self._statechart.note_pressure()
            return True
        return False

    async def _store_payload_task(
        self,
        *,
        payload: str,
        user_id: str,
        metadata: dict[str, Any],
    ) -> None:
        await asyncio.to_thread(self._store_semaphore.acquire)
        self._store_inflight += 1
        try:
            result = await self.memory_tools.store_memory(
                content=payload,
                user_id=user_id,
                metadata=metadata,
            )
        finally:
            self._store_inflight -= 1
            self._store_semaphore.release()
        if isinstance(result, Failure):
            self._statechart.note_failure()
            raise TypeError(str(result.failure()))
        self._statechart.note_success()

    async def recall(self, *, question: str, user_id: str, memory_scope: str) -> RecallResult:
        if self._statechart.mode == 'shed':
            return RecallResult(text='No previous statements found.', count=0, status='shed')

        recall_timeout_seconds = max(0.1, self.config.recall_timeout_ms / 1000.0)
        try:
            result = await asyncio.wait_for(
                self.memory_tools.search_memories(
                    query=question,
                    user_id=user_id,
                    filters={'memory_scope': memory_scope},
                ),
                timeout=recall_timeout_seconds,
            )
        except TimeoutError:
            # ``TimeoutError.__str__`` is empty by default, which made earlier
            # warnings show only ``"recall failed (healthy): "``; spell it out.
            logger.warning(
                f'[MemoryOrchestrator]: recall timed out after '
                f'{recall_timeout_seconds:.2f}s (mode={self.mode}).'
            )
            self._statechart.note_failure()
            return RecallResult(text='No previous statements found.', count=0, status='timeout')
        except Exception as error:
            logger.warning(
                f'[MemoryOrchestrator]: recall failed ({self.mode}): '
                f'{type(error).__name__}: {error}'
            )
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
                logger.warning(
                    f'[MemoryOrchestrator]: recall backend failure ({self.mode}): {error}'
                )
                self._statechart.note_failure()
                return RecallResult(text='No previous statements found.', count=0, status='error')
            case _:
                self._statechart.note_failure()
                return RecallResult(text='No previous statements found.', count=0, status='unknown')

    async def store(
        self,
        *,
        question: str,
        answer: str,
        reasoning: str,
        user_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist one memory entry under the configured timeout and bulkhead.

        The write is always awaited within the caller's lifetime — a
        prior fire-and-forget variant would be orphaned by
        :func:`dspy.syncify`'s per-call event loops.  Failures update
        the statechart through ``_store_payload_task`` but are swallowed
        here so a transient backend hiccup does not break the agent
        turn.
        """
        if self._should_skip_store():
            return

        payload = self.memory_tools.format_store_memory_text(
            question=question,
            answer=answer,
            reasoning=reasoning,
        )
        store_timeout_seconds = max(0.1, self.config.store_timeout_ms / 1000.0)
        try:
            await asyncio.wait_for(
                self._store_payload_task(
                    payload=payload,
                    user_id=user_id,
                    metadata=metadata,
                ),
                timeout=store_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                f'[MemoryOrchestrator]: store timed out after '
                f'{store_timeout_seconds:.2f}s (mode={self.mode}).'
            )
            self._statechart.note_failure()
        except Exception as error:
            # _store_payload_task already records statechart success/failure on
            # the backend-error path; this catch is the safety net for anything
            # that escapes (e.g. cancellation, slot acquisition errors).
            logger.warning(
                f'[MemoryOrchestrator]: store failed ({self.mode}): {type(error).__name__}: {error}'
            )
