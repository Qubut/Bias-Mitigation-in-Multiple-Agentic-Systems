"""mem0.AsyncMemory wrapper.

Recovery: tenacity retries, purgatory breakers, one-shot dimension
fallback, optional resilient empty.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from itertools import accumulate, chain
from threading import BoundedSemaphore
from typing import Any, cast

from asyncstdlib.functools import lru_cache as alru_cache
from loguru import logger
from mem0 import AsyncMemory
from purgatory import AsyncCircuitBreakerFactory
from purgatory.domain.model import OpenedState
from purgatory.service._async.circuitbreaker import AsyncCircuitBreaker
from returns.functions import tap
from returns.iterables import Fold
from returns.pipeline import flow
from returns.pointfree import cond
from returns.result import Failure, Result, Success
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from bias_mitigation.data.models.memory_config import Mem0Config
from bias_mitigation.memory.contracts import MemoryRecord, MemorySearchResult
from bias_mitigation.memory.errors import (
    MemoryContractError,
    MemorySearchError,
    MemoryStoreError,
    MemoryToolError,
)
from bias_mitigation.memory.mem0_compat import disable_mem0_telemetry, patch_openai_embedder

_CANCEL_ENV_VAR = 'BIAS_MITIGATION_CANCEL_REQUESTED'

_RECOVERABLE_ERROR_MARKERS = (
    'unterminated string',
    'new_retrieved_facts',
    'out of bounds',
    'index',
    'json',
    'decode',
    'operands could not be broadcast together',
    'broadcast together with shapes',
    'bad parameter or other api misuse',
    'expecting',
    'delimiter',
)

_DIMENSION_ERROR_MARKERS = (
    'dimension',
    'dimensions',
    'matryoshka',
    'embedding_model_dims',
    'size mismatch',
    'expected dim',
)


class _MemoryBackpressureError(RuntimeError):
    """Slot saturated or pressure breaker open. Caught inside Mem0Tools."""


class Mem0Tools:
    """mem0.AsyncMemory + retry/breakers/fallbacks.

    Backend is built in __init__; bad config blows up here, not later.
    self.memory can be rebuilt at runtime when the dim fallback fires.
    """

    def __init__(self, config: Mem0Config) -> None:
        self.config = config
        self._using_dimensionless_config = False
        # threading.BoundedSemaphore, not asyncio — dspy.syncify gives each
        # worker its own event loop, and asyncio.Semaphore is loop-local
        # (either raises "bound to a different event loop" or hangs and
        # trips the pressure breaker). asyncio.to_thread bridges the
        # blocking acquire down in _memory_slot.
        self._memory_op_semaphore = BoundedSemaphore(
            self.config.memory_operation_semaphore_limit,
        )
        self._events: Counter[str] = Counter()

        self._breaker_factory = AsyncCircuitBreakerFactory()

        logger.info('[Mem0Tools]: Initializing Mem0')
        patch_openai_embedder(force_dimensionless=self.config.embedder_force_dimensionless_requests)
        disable_mem0_telemetry()
        self.memory: AsyncMemory = AsyncMemory.from_config(self.config.to_mem0_dict(False))

    def format_store_memory_text(self, *, question: str, answer: str, reasoning: str) -> str:
        """Format the `Question | Answer | Reasoning` payload to store.

        Drop the question via `include_question_in_memory_text: false`
        if you don't want sensitive attributes leaking back on recall.
        """
        parts = [
            f'Question: {self._normalize_text(question)}'
            if self.config.include_question_in_memory_text
            else '',
            f'Answer: {self._normalize_text(answer)}',
            f'Reasoning: {self._normalize_text(reasoning)}',
        ]
        return self._sanitize_store_content(' | '.join(p for p in parts if p))

    def render_recalled_memory_text(self, passages: list[str]) -> str:
        """Format recalled passages for a prompt.

        Dedupes, normalises, peels off legacy JSON wrappers, caps
        count/chars. Empty after all that → 'No previous statements found.'.
        """
        cleaned = self._semantic_recalled_passages(passages)
        if not cleaned:
            return 'No previous statements found.'
        template = '{}' if self.config.render_recalled_memory_style == 'plain' else '- {}'
        return '\n'.join(template.format(snippet) for snippet in cleaned)

    @alru_cache(maxsize=None)
    async def _pressure_breaker(self) -> AsyncCircuitBreaker:
        return await self._breaker_factory.get_breaker(
            'pressure',
            threshold=self.config.pressure_timeout_trip_threshold,
            ttl=self.config.pressure_cooldown_ms / 1000.0,
        )

    @alru_cache(maxsize=None)
    async def _search_fallback_breaker(self) -> AsyncCircuitBreaker:
        return await self._breaker_factory.get_breaker(
            'search_fallback',
            threshold=self.config.search_fallback_consecutive_fail_trip_threshold,
            ttl=self.config.search_fallback_cooldown_ms / 1000.0,
        )

    async def search_memories(
        self,
        query: str | list[str],
        user_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Result[MemorySearchResult, MemoryToolError]:
        """Recall passages. Pass a list to fan out and merge.

        Failure comes back only when resilient mode is off and every
        recovery branch has been used up.
        """
        self._raise_if_cancelled('search')
        queries = [query] if isinstance(query, str) else query
        search_args = self._build_search_args(
            user_id=user_id,
            limit=limit or self.config.recall_top_k,
            filters=filters,
        )
        per_query: list[Result[list[MemoryRecord], MemoryToolError]] = [
            await self._search_with_recovery(q, search_args) for q in queries
        ]
        seed: Result[tuple[list[MemoryRecord], ...], MemoryToolError] = Success(())
        collected = cast(
            Result[tuple[list[MemoryRecord], ...], MemoryToolError],
            Fold.collect(per_query, seed),
        )
        return collected.map(self._merge_search_passages).lash(self._maybe_resilient_empty_search)

    def _merge_search_passages(
        self, item_lists: tuple[list[MemoryRecord], ...]
    ) -> MemorySearchResult:
        all_items = list(chain.from_iterable(item_lists))
        passages = self._semantic_recalled_passages(self._extract_passages(all_items))
        return {'passages': passages, 'count': len(passages)}

    def _maybe_resilient_empty_search(
        self, error: MemoryToolError
    ) -> Result[MemorySearchResult, MemoryToolError]:
        empty: MemorySearchResult = {'passages': [], 'count': 0}
        return cast(
            Result[MemorySearchResult, MemoryToolError],
            cond(Result, empty, error)(self.config.enable_resilient_search_fallback).map(
                tap(lambda _: self._emit('search.graceful_empty'))
            ),
        )

    async def store_memory(
        self,
        content: str | list[dict[str, Any]],
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result[None, MemoryToolError]:
        """Persist content. Prefer pre-formatting via format_store_memory_text.

        Goes through pressure-mode gate, dim fallback, infer=False retry
        on parse errors.
        """
        self._raise_if_cancelled('store')
        self._emit('store.attempts')

        if self.config.memory_pressure_mode in {'disabled', 'read_only'}:
            self._emit('store.skipped_pressure_mode')
            return Success(None)

        normalized = self._sanitize_store_content(
            content if isinstance(content, str) else str(content),
        )
        if not normalized:
            self._emit('store.skipped_empty_content')
            return Success(None)

        return await self._store_with_recovery(
            content=normalized,
            user_id=user_id,
            metadata=metadata or {},
        )

    def stats_snapshot(self) -> dict[str, int]:
        """Event counters + using_dimensionless_config flag, for post-mortem."""
        return {
            **dict(self._events),
            'using_dimensionless_config': int(self._using_dimensionless_config),
        }

    async def _search_with_recovery(
        self, query: str, search_args: dict[str, Any]
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        self._emit('search.attempts')
        normalized_query = self._sanitize_query(query)
        if not normalized_query:
            self._emit('search.graceful_empty')
            return Success([])
        raw = await self._raw_search(normalized_query, search_args)
        transformed = raw.bind(self._normalize_search_payload).map(
            tap(lambda _: self._emit('search.success'))
        )
        if isinstance(transformed, Failure):
            return await self._handle_search_error(
                normalized_query,
                search_args,
                transformed.failure(),
            )
        return cast(Result[list[MemoryRecord], MemoryToolError], transformed)

    async def _handle_search_error(
        self, query: str, search_args: dict[str, Any], error: Exception
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        match error:
            case _MemoryBackpressureError():
                return self._on_search_backpressure()
            case err if self._attempt_dimension_fallback_reinit(err):
                return await self._retry_search_after_dim_reinit(query, search_args)
            case err if self._is_recoverable_memory_error(err):
                self._emit('search.recoverable_failures')
                return await self._fallback_search(query, search_args)
            case err:
                self._emit('search.failures')
                return Failure(
                    MemorySearchError(message='Mem0 search operation failed.', cause=err)
                )

    async def _retry_search_after_dim_reinit(
        self, query: str, search_args: dict[str, Any]
    ) -> Result[list[MemoryRecord], MemoryToolError]:

        def _on_retry_success(_payload: list[MemoryRecord]) -> None:
            self._emit('search.success')
            self._emit('search.dimension_fallback_retry_success')

        raw = await self._raw_search(query, search_args)
        transformed = raw.bind(self._normalize_search_payload).map(tap(_on_retry_success))
        if isinstance(transformed, Failure):
            return await self._handle_search_error(query, search_args, transformed.failure())
        return cast(Result[list[MemoryRecord], MemoryToolError], transformed)

    async def _raw_search(self, query: str, search_args: dict[str, Any]) -> Result[Any, Exception]:
        try:
            payload = await self._run_with_retries(
                attempts=self.config.search_retry_attempts,
                event='search.transient_retries',
                operation=lambda: self.memory.search(query=query, **search_args),
            )
            return Success(payload)
        except Exception as error:
            return Failure(error)

    def _on_search_backpressure(self) -> Result[list[MemoryRecord], MemoryToolError]:
        error: MemoryToolError = MemorySearchError(
            message='Mem0 search blocked by pressure backoff policy.',
            cause=_MemoryBackpressureError('Mem0 search backpressure saturation.'),
        )
        empty: list[MemoryRecord] = []
        return cast(
            Result[list[MemoryRecord], MemoryToolError],
            cond(Result, empty, error)(self.config.degrade_search_on_backpressure)
            .map(tap(lambda _: self._note_search_degraded()))
            .alt(tap(lambda _: self._emit('search.failures'))),
        )

    def _note_search_degraded(self) -> None:
        self._emit('search.backpressure_degrades')
        self._emit('search.graceful_empty')
        self._pressure_warn('search')

    async def _fallback_search(
        self, query: str, search_args: dict[str, Any]
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        self._emit('search.fallback_attempts')
        fallback_args = {**search_args, 'top_k': self.config.search_fallback_limit}
        attempts = max(1, int(getattr(self.config, 'search_fallback_retry_attempts', 1)))
        breaker = await self._search_fallback_breaker()
        try:
            async with breaker:
                payload = await self._run_with_retries(
                    attempts=attempts,
                    event='search.transient_retries',
                    operation=lambda: self.memory.search(query=query, **fallback_args),
                )
        except OpenedState:
            self._emit('search.fallback_circuit_open_skips')
            self._emit('search.graceful_empty')
            self._maybe_log_fallback_skip()
            return Success([])
        except Exception as fallback_error:
            return self._on_fallback_search_failure(
                fallback_error,
                just_tripped=breaker.context.state == 'opened',
            )

        self._emit('search.fallback_success')
        self._emit('search.success')
        return self._normalize_search_payload(payload)

    def _on_fallback_search_failure(
        self,
        error: Exception,
        *,
        just_tripped: bool,
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        wrapped: MemoryToolError = MemorySearchError(
            message='Mem0 search fallback failed.', cause=error
        )
        empty: list[MemoryRecord] = []
        return cast(
            Result[list[MemoryRecord], MemoryToolError],
            cond(Result, empty, wrapped)(self.config.enable_resilient_search_fallback).map(
                tap(lambda _: self._note_fallback_failure_degraded(just_tripped=just_tripped))
            ),
        )

    def _note_fallback_failure_degraded(self, *, just_tripped: bool) -> None:
        self._emit('search.graceful_empty')
        self._maybe_log_fallback_failure(just_tripped)

    def _maybe_log_fallback_skip(self) -> None:
        skips = self._events['search.fallback_circuit_open_skips']
        if skips == 1 or skips % self.config.search_fallback_warning_every == 0:
            self._emit('search.fallback_warning_emitted')
            logger.warning(
                f'[Mem0Tools]: Search fallback suppression active; returning empty '
                f'recalled memory (skip_count={skips}).'
            )

    def _maybe_log_fallback_failure(self, tripped: bool) -> None:
        graceful = self._events['search.graceful_empty']
        every = self.config.search_fallback_warning_every
        if not (graceful == 1 or graceful % every == 0 or tripped):
            return
        self._emit('search.fallback_warning_emitted')
        if tripped:
            self._emit('search.fallback_circuit_open_events')
            logger.warning(
                f'[Mem0Tools]: Recoverable search fallback repeatedly failed; '
                f'opening fallback suppression circuit and continuing with empty '
                f'recalled memory (count={graceful}).'
            )
        else:
            logger.warning(
                f'[Mem0Tools]: Recoverable search fallback failed; continuing with '
                f'empty recalled memory (count={graceful}).'
            )

    async def _store_with_recovery(
        self, *, content: str, user_id: str | None, metadata: dict[str, Any]
    ) -> Result[None, MemoryToolError]:

        async def _try(infer: bool) -> Result[None, Exception]:
            return await self._raw_store_add(content, user_id, metadata, infer=infer)

        primary = await _try(self.config.memory_add_infer)
        if isinstance(primary, Success):
            self._emit('store.success')
            return Success(None)

        match primary.failure():
            case _MemoryBackpressureError():
                return self._on_store_backpressure()
            case err if self._attempt_dimension_fallback_reinit(err):
                return self._classify_store_retry(
                    await _try(self.config.memory_add_infer),
                    success_event='store.dimension_fallback_retry_success',
                )
            case err if self._is_recoverable_memory_error(err) and self.config.memory_add_infer:
                self._emit('store.infer_false_retries')
                return self._classify_store_retry(
                    await _try(infer=False),
                    success_event='store.infer_false_retry_success',
                )
            case err if self._is_recoverable_memory_error(err):
                self._emit('store.recoverable_failures')
                return Success(None)
            case err:
                return self._store_failure(err)

    def _classify_store_retry(
        self, outcome: Result[None, Exception], *, success_event: str
    ) -> Result[None, MemoryToolError]:
        if isinstance(outcome, Success):
            self._emit('store.success')
            self._emit(success_event)
            return Success(None)
        err = outcome.failure()
        if self._is_recoverable_memory_error(err):
            self._emit('store.recoverable_failures')
            return Success(None)
        return self._store_failure(err)

    async def _raw_store_add(
        self, content: str, user_id: str | None, metadata: dict[str, Any], *, infer: bool
    ) -> Result[None, Exception]:
        try:
            await self._run_with_retries(
                attempts=self.config.store_retry_attempts,
                event='store.transient_retries',
                operation=lambda: self.memory.add(
                    content, user_id=user_id, metadata=metadata, infer=infer
                ),
            )
            return Success(None)
        except Exception as error:
            return Failure(error)

    def _on_store_backpressure(self) -> Result[None, MemoryToolError]:
        error: MemoryToolError = MemoryStoreError(
            message='Mem0 add operation failed while storing memory.',
            cause=_MemoryBackpressureError('Mem0 store backpressure saturation.'),
        )
        return cast(
            Result[None, MemoryToolError],
            cond(Result, None, error)(self.config.drop_store_on_backpressure)
            .map(tap(lambda _: self._note_store_dropped()))
            .alt(tap(lambda _: self._emit('store.failures'))),
        )

    def _note_store_dropped(self) -> None:
        self._emit('store.backpressure_skips')
        self._pressure_warn('store')

    def _store_failure(self, error: Exception) -> Result[None, MemoryToolError]:
        self._emit('store.failures')
        return Failure(
            MemoryStoreError(message='Mem0 add operation failed while storing memory.', cause=error)
        )

    async def _run_with_retries(
        self,
        *,
        attempts: int,
        event: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run `operation` with tenacity retry + slot acquisition.

        Only `_is_transient_backend_error` triggers a retry. Everything
        else propagates so the recovery layer can classify it.
        """

        def before_sleep(_state: Any) -> None:
            self._raise_if_cancelled('retry')
            self._emit(event)

        retryer = AsyncRetrying(
            stop=stop_after_attempt(max(1, attempts)),
            wait=wait_exponential_jitter(
                initial=max(self.config.retry_backoff_min_ms / 1000.0, 0.001),
                max=max(self.config.retry_backoff_max_ms / 1000.0, 0.001),
                jitter=max(self.config.retry_jitter_ms / 1000.0, 0.0),
            ),
            retry=retry_if_exception(self._is_transient_backend_error),
            before_sleep=before_sleep,
            reraise=True,
        )

        async def run_once() -> Any:
            self._raise_if_cancelled('memory_op')
            async with self._memory_slot():
                return await operation()

        return await retryer(run_once)

    @asynccontextmanager
    async def _memory_slot(self) -> AsyncIterator[None]:
        """Bounded slot behind the pressure breaker.

        Cap = config.memory_operation_semaphore_limit, kept low so
        parallel agents don't blow past the embedding provider's rate
        limit. Open breaker or timed-out acquire → `_MemoryBackpressureError`.
        """
        self._raise_if_cancelled('memory_slot')
        timeout_seconds = max(0.01, self.config.memory_slot_timeout_ms / 1000.0)
        try:
            async with await self._pressure_breaker():
                acquired = await asyncio.to_thread(
                    self._memory_op_semaphore.acquire,
                    True,
                    timeout_seconds,
                )
                if not acquired:
                    self._raise_if_cancelled('memory_slot_wait')
                    self._emit('semaphore.wait_timeouts')
                    raise _MemoryBackpressureError('Memory operation slot saturated.')
        except OpenedState as exc:
            self._emit('pressure.circuit_open_skips')
            raise _MemoryBackpressureError(
                'Memory operation rejected while pressure circuit is open.'
            ) from exc
        try:
            yield
        finally:
            self._memory_op_semaphore.release()

    def _pressure_warn(self, operation: str) -> None:
        skips = self._events['pressure.circuit_open_skips']
        if skips == 1 or skips % self.config.pressure_warning_every == 0:
            self._emit('pressure.warning_emitted')
            logger.warning(
                f'[Mem0Tools]: Pressure circuit open; {operation} degraded (skip_count={skips}).'
            )

    def _attempt_dimension_fallback_reinit(self, error: Exception) -> bool:
        """Rebuild mem0 without the explicit embedding dim. Fires at most once.

        Matryoshka embedders sometimes accept a target dimension but
        mismatch the vector store at query time. Returns True if the
        rebuild took and the caller should retry.
        """
        if self._using_dimensionless_config:
            return False
        if not self._supports_dimension_fallback():
            return False
        if not self._is_dimension_related_error(error):
            return False

        logger.warning(
            '[Mem0Tools]: Detected dimension-related Mem0 error. '
            'Retrying with embedding dimension override removed.'
        )
        try:
            self.memory = AsyncMemory.from_config(self.config.to_mem0_dict(True))
        except Exception as reinit_error:
            logger.error(f'[Mem0Tools]: Dimension fallback initialization failed: {reinit_error}')
            return False
        self._using_dimensionless_config = True
        self._emit('dimension_fallback.activations')
        return True

    def _supports_dimension_fallback(self) -> bool:
        if self.config.vector_store is None:
            return False
        return bool(
            self.config.enable_dimension_fallback
            and self.config.vector_store.config.embedding_model_dims
        )

    @staticmethod
    def _is_dimension_related_error(error: Exception) -> bool:
        text = str(error).lower()
        return any(marker in text for marker in _DIMENSION_ERROR_MARKERS)

    @staticmethod
    def _is_recoverable_memory_error(error: Exception) -> bool:
        text = str(error).lower()
        return any(marker in text for marker in _RECOVERABLE_ERROR_MARKERS)

    def _is_transient_backend_error(self, error: BaseException) -> bool:
        text = str(error).lower()
        return any(marker.lower() in text for marker in self.config.transient_error_markers)

    @staticmethod
    def _build_search_args(
        user_id: str | None, limit: int, filters: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build mem0 2.x search kwargs.

        mem0 2.0 expects entity IDs inside `filters` and uses `top_k`
        not `limit`. IDs at the top level raise `Mem0ValidationError`,
        and resilient mode would eat that as a graceful empty so the
        bug never surfaces. Centralise the translation here.
        """
        combined_filters: dict[str, Any] = dict(filters) if filters else {}
        if user_id:
            combined_filters['user_id'] = user_id
        args: dict[str, Any] = {'top_k': limit}
        if combined_filters:
            args['filters'] = combined_filters
        return args

    @staticmethod
    def _normalize_search_payload(
        payload: Any,
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        """Unpack mem0's {'results': [...]} envelope."""
        if isinstance(payload, dict) and isinstance(payload.get('results'), list):
            return Success([
                cast(MemoryRecord, item) for item in payload['results'] if isinstance(item, dict)
            ])
        return Failure(
            MemoryContractError(message='Mem0 search payload is missing list-like `results`.')
        )

    @staticmethod
    def _extract_passages(items: list[MemoryRecord]) -> list[str]:
        return [text for item in items if (text := item.get('memory') or '')]

    def _semantic_recalled_passages(self, passages: list[str]) -> list[str]:
        return flow(
            passages,
            self._strip_empty_normalized,
            self._dedupe_preserving_order,
            self._cap_recalled_items,
            self._normalize_passages,
        )

    def _cap_recalled_items(self, items: list[str]) -> list[str]:
        return items[: self.config.recall_max_items]

    def _strip_empty_normalized(self, texts: list[str]) -> list[str]:
        return [normalized for t in texts if (normalized := self._normalize_text(t))]

    @staticmethod
    def _dedupe_preserving_order(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    def _normalize_passages(self, passages: list[str]) -> list[str]:
        candidates: list[str] = [
            normalised
            for passage in passages
            if (
                normalised := self._truncate(
                    self._normalize_text(passage), self.config.max_passage_chars
                )
            )
        ]
        cap = self.config.max_recalled_chars
        prev_totals = accumulate((len(c) for c in candidates), initial=0)
        return [
            clipped
            for candidate, prev_total in zip(candidates, prev_totals, strict=False)
            if prev_total < cap
            if (clipped := candidate[: cap - prev_total].rstrip())
        ]

    def _emit(self, event: str) -> None:
        self._events[event] += 1

    @staticmethod
    def _raise_if_cancelled(stage: str) -> None:
        if os.getenv(_CANCEL_ENV_VAR, '0') == '1':
            raise KeyboardInterrupt(f'Memory operation cancelled ({stage}).')

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace('\x00', ' ').strip()

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        return text if len(text) <= max_chars else text[:max_chars].rstrip()

    def _sanitize_query(self, query: str) -> str:
        return self._truncate(self._normalize_text(query), self.config.max_query_chars)

    def _sanitize_store_content(self, content: str) -> str:
        return self._truncate(self._normalize_text(content), self.config.max_memory_content_chars)
