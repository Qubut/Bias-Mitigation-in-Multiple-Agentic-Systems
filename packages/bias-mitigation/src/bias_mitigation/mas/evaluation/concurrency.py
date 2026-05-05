"""Concurrency bulkheads for deterministic evaluator execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

from loguru import logger


@dataclass(frozen=True, slots=True)
class ConcurrencyConfig:
    """Typed concurrency controls for deterministic evaluation."""

    max_evaluation_threads: int
    enable_monitoring: bool = True


class ConcurrencyManager:
    """Executor + semaphore bulkheads to prevent thread oversubscription."""

    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        max_workers = max(1, int(config.max_evaluation_threads))
        self._max_workers = max_workers
        self._eval_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='MAS_Eval',
        )
        self._submission_semaphore = asyncio.Semaphore(max_workers)
        self._inflight_lock = Lock()
        self._inflight_submissions = 0
        logger.info(
            '[ConcurrencyManager]: initialized '
            f'(max_evaluation_threads={max_workers}).'
        )

    @property
    def max_evaluation_threads(self) -> int:
        return max(1, int(self.config.max_evaluation_threads))

    async def run_in_eval_pool(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run blocking work in bounded executor with in-flight submission bulkhead.

        Important: capacity is released only when the underlying executor future
        completes, not when the awaiting coroutine is cancelled/times out.
        This prevents queue growth under repeated per-task timeouts.
        """
        await self._submission_semaphore.acquire()
        loop = asyncio.get_running_loop()
        with self._inflight_lock:
            self._inflight_submissions += 1
            inflight_now = self._inflight_submissions

        if self.config.enable_monitoring:
            logger.debug(
                '[ConcurrencyManager]: scheduling eval task '
                f'(eval_pool_inflight={inflight_now}/{self._max_workers}).'
            )

        executor_future = loop.run_in_executor(self._eval_executor, fn, *args)

        def _release_capacity(_future: Any) -> None:
            def _finalize_release() -> None:
                with self._inflight_lock:
                    self._inflight_submissions = max(0, self._inflight_submissions - 1)
                self._submission_semaphore.release()

            loop.call_soon_threadsafe(_finalize_release)

        executor_future.add_done_callback(_release_capacity)

        return await asyncio.shield(executor_future)

    def shutdown(self) -> None:
        """Release executor resources at evaluation end."""
        self._eval_executor.shutdown(wait=True, cancel_futures=False)
        logger.info('[ConcurrencyManager]: shutdown complete.')
