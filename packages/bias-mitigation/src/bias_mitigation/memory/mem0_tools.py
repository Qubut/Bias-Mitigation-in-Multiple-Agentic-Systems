"""Declarative Mem0 tooling wrapper using typed Result flows."""

import json
import os
import time
from contextlib import contextmanager
from itertools import chain
from threading import BoundedSemaphore
from typing import Any, cast

# Ensure Mem0/PostHog telemetry is disabled before mem0 package import-time side effects.
os.environ['MEM0_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLED'] = 'true'
os.environ['DO_NOT_TRACK'] = 'true'
os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'

from loguru import logger
from mem0 import Memory
from mem0.memory import main as mem0_main
from mem0.memory import telemetry as mem0_telemetry
from returns.result import Failure, Result, Success
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from bias_mitigation.data.models.memory_config import Mem0Config
from bias_mitigation.memory.contracts import (
    MemoryProviderProtocol,
    MemoryRecord,
    MemorySearchResult,
)
from bias_mitigation.memory.errors import (
    MemoryClearError,
    MemoryConfigurationError,
    MemoryContractError,
    MemorySearchError,
    MemoryStoreError,
    MemoryToolError,
)

try:
    from mem0.embeddings.openai import OpenAIEmbedding
except Exception:
    OpenAIEmbedding = None


class _MemoryBackpressureError(RuntimeError):
    """Raised when memory operation slots are saturated and pressure mode is active."""


class Mem0Tools:
    """Adapter/repository abstraction around Mem0 operations.

    Architecture:
    - Pure helpers: argument building + payload normalization + projection.
    - Effectful gateways: provider `add/search/get_all/delete` calls only.
    """

    _CANCEL_ENV_VAR = 'BIAS_MITIGATION_CANCEL_REQUESTED'
    _EMBEDDER_PATCH_VERSION = 2
    _FORCE_DIMENSIONLESS_EMBED_REQUESTS = True

    @staticmethod
    def _embed_without_dimensions(instance: Any, text: str) -> list[float]:
        normalized_text = text.replace('\n', ' ')
        response = instance.client.embeddings.create(
            input=[normalized_text],
            model=instance.config.model,
        )
        return cast(list[float], response.data[0].embedding)

    @classmethod
    def _build_patched_openai_embed(cls, original_embed: Any) -> Any:
        force_dimensionless_ids: set[int] = set()

        def _patched_embed(
            instance: Any, text: str, memory_action: str | None = None
        ) -> list[float]:
            instance_id = id(instance)
            should_force_dimensionless = (
                cls._FORCE_DIMENSIONLESS_EMBED_REQUESTS or instance_id in force_dimensionless_ids
            )
            if should_force_dimensionless:
                return cls._embed_without_dimensions(instance, text)

            try:
                return cast(list[float], original_embed(instance, text, memory_action))
            except Exception as error:
                error_text = str(error).lower()
                if 'matryoshka' not in error_text and 'dimension' not in error_text:
                    raise

                force_dimensionless_ids.add(instance_id)
                logger.warning(
                    '[Mem0Tools]: Embedder endpoint rejected dimensions. '
                    'Switching this embedder instance to dimensionless requests.'
                )
                return cls._embed_without_dimensions(instance, text)

        return _patched_embed

    def __init__(self, config: Mem0Config):
        self.config = config
        Mem0Tools._FORCE_DIMENSIONLESS_EMBED_REQUESTS = (
            self.config.embedder_force_dimensionless_requests
        )
        self._using_dimensionless_config = False
        self._memory_op_semaphore = BoundedSemaphore(self.config.memory_operation_semaphore_limit)
        self._pressure_open_until_monotonic = 0.0
        self._consecutive_slot_timeouts = 0
        self._search_fallback_open_until_monotonic = 0.0
        self._search_fallback_consecutive_failures = 0
        self._stats: dict[str, int] = {
            'store_attempts': 0,
            'store_success': 0,
            'store_failures': 0,
            'store_recoverable_failures': 0,
            'store_infer_false_retries': 0,
            'store_infer_false_retry_success': 0,
            'store_skipped_empty_content': 0,
            'store_skipped_pressure_mode': 0,
            'store_transient_retries': 0,
            'search_attempts': 0,
            'search_success': 0,
            'search_failures': 0,
            'search_recoverable_failures': 0,
            'search_fallback_attempts': 0,
            'search_fallback_success': 0,
            'search_fallback_circuit_open_events': 0,
            'search_fallback_circuit_open_skips': 0,
            'search_fallback_consecutive_failures': 0,
            'search_graceful_empty': 0,
            'search_fallback_warning_emitted': 0,
            'search_transient_retries': 0,
            'clear_attempts': 0,
            'clear_success': 0,
            'clear_failures': 0,
            'dimension_fallback_activations': 0,
            'dimension_fallback_retry_success': 0,
            'semaphore_wait_timeouts': 0,
            'pressure_circuit_open_events': 0,
            'pressure_circuit_open_skips': 0,
            'store_backpressure_skips': 0,
            'search_backpressure_degrades': 0,
            'pressure_warning_emitted': 0,
        }
        logger.info('[Mem0Tools]: Initializing Mem0')
        self._apply_mem0_openai_embedder_compat_patch()
        self._disable_mem0_telemetry_hooks()
        self.memory = self._initialize_provider(strip_embedding_dims=False)

    @staticmethod
    def _build_noop_posthog_type() -> type:
        class _NoOpPosthog:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs
                self.disabled = True

            def capture(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

            def shutdown(self) -> None:
                return

        return _NoOpPosthog

    @staticmethod
    def _noop_capture_event(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _patch_mem0_posthog(self, no_op_posthog_type: type) -> None:
        mem0_telemetry.Posthog = no_op_posthog_type
        client = getattr(mem0_telemetry, 'client_telemetry', None)
        if client is not None:
            client.posthog = no_op_posthog_type()

    def _patch_mem0_capture_hooks(self) -> None:
        mem0_telemetry.capture_event = self._noop_capture_event
        capture_client_event = getattr(mem0_telemetry, 'capture_client_event', None)
        if capture_client_event is not None:
            mem0_telemetry.capture_client_event = self._noop_capture_event
        mem0_main.capture_event = self._noop_capture_event

        search_globals = getattr(Memory.search, '__globals__', None)
        if isinstance(search_globals, dict) and 'capture_event' in search_globals:
            search_globals['capture_event'] = self._noop_capture_event

    def _disable_mem0_telemetry_hooks(self) -> None:
        try:
            no_op_posthog_type = self._build_noop_posthog_type()
            self._patch_mem0_posthog(no_op_posthog_type)
            self._patch_mem0_capture_hooks()
            logger.info('[Mem0Tools]: Disabled Mem0 telemetry capture hooks.')
        except Exception as error:
            logger.debug(f'[Mem0Tools]: Unable to patch Mem0 telemetry hooks: {error}')

    def _apply_mem0_openai_embedder_compat_patch(self) -> None:
        """Patch Mem0 OpenAI embedder to gracefully retry without dimensions.

        Some OpenAI-compatible embedding endpoints reject explicit `dimensions`
        (e.g., with matryoshka-related 400 errors). Mem0 currently always sends
        this parameter for its OpenAI embedder implementation.
        """
        if OpenAIEmbedding is None:
            logger.debug('[Mem0Tools]: Skipping embedder compatibility patch: embedder unavailable')
            return

        if (
            getattr(OpenAIEmbedding, 'bias_mitigation_dimensions_patch_version', 0)
            >= self._EMBEDDER_PATCH_VERSION
        ):
            return

        original_embed = OpenAIEmbedding.embed
        OpenAIEmbedding.embed = self._build_patched_openai_embed(original_embed)
        OpenAIEmbedding.bias_mitigation_dimensions_patch_applied = True
        OpenAIEmbedding.bias_mitigation_dimensions_patch_version = self._EMBEDDER_PATCH_VERSION

    def _initialize_provider(self, strip_embedding_dims: bool) -> MemoryProviderProtocol:
        try:
            provider = Memory.from_config(self.config.to_mem0_dict(strip_embedding_dims))
            return cast(MemoryProviderProtocol, provider)
        except Exception as error:
            raise MemoryConfigurationError(
                message='Failed to initialize Mem0 provider from configuration.',
                cause=error,
            ) from error

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace('\x00', ' ').strip()

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    def _sanitize_query(self, query: str) -> str:
        normalized = self._normalize_text(query)
        return self._truncate(normalized, self.config.max_query_chars)

    def _sanitize_store_content(self, content: str) -> str:
        normalized = self._normalize_text(content)
        return self._truncate(normalized, self.config.max_memory_content_chars)

    @classmethod
    def _is_cancel_requested(cls) -> bool:
        return os.getenv(cls._CANCEL_ENV_VAR, '0') == '1'

    def _is_pressure_open(self) -> bool:
        return time.monotonic() < self._pressure_open_until_monotonic

    def _record_pressure_skip(self) -> None:
        self._stats['pressure_circuit_open_skips'] += 1

    def _pressure_warn(self, operation: str) -> None:
        total_skips = self._stats['pressure_circuit_open_skips']
        should_warn = total_skips == 1 or total_skips % self.config.pressure_warning_every == 0
        if not should_warn:
            return
        self._stats['pressure_warning_emitted'] += 1
        logger.warning(
            f'[Mem0Tools]: Pressure circuit open; {operation} degraded (skip_count={total_skips}).'
        )

    def _open_pressure_circuit(self) -> None:
        cooldown_seconds = max(self.config.pressure_cooldown_ms / 1000.0, 0.1)
        self._pressure_open_until_monotonic = time.monotonic() + cooldown_seconds
        self._consecutive_slot_timeouts = 0
        self._stats['pressure_circuit_open_events'] += 1

    def _record_slot_timeout(self) -> None:
        self._stats['semaphore_wait_timeouts'] += 1
        self._consecutive_slot_timeouts += 1
        if self._consecutive_slot_timeouts >= self.config.pressure_timeout_trip_threshold:
            self._open_pressure_circuit()

    def _record_slot_success(self) -> None:
        self._consecutive_slot_timeouts = 0

    def _search_fallback_retry_attempts(self) -> int:
        configured = getattr(self.config, 'search_fallback_retry_attempts', 1)
        return max(1, int(configured))

    def _search_fallback_trip_threshold(self) -> int:
        configured = getattr(
            self.config,
            'search_fallback_consecutive_fail_trip_threshold',
            8,
        )
        return max(1, int(configured))

    def _search_fallback_cooldown_seconds(self) -> float:
        configured_ms = getattr(self.config, 'search_fallback_cooldown_ms', 20_000)
        try:
            normalized_ms = int(configured_ms)
        except Exception:
            normalized_ms = 20_000
        return max(normalized_ms / 1000.0, 0.1)

    def _is_search_fallback_open(self) -> bool:
        return time.monotonic() < self._search_fallback_open_until_monotonic

    def _open_search_fallback_circuit(self) -> None:
        cooldown_seconds = self._search_fallback_cooldown_seconds()
        self._search_fallback_open_until_monotonic = time.monotonic() + cooldown_seconds
        self._search_fallback_consecutive_failures = 0
        self._stats['search_fallback_circuit_open_events'] += 1

    def _record_search_fallback_success(self) -> None:
        self._search_fallback_consecutive_failures = 0
        self._stats['search_fallback_consecutive_failures'] = 0

    def _record_search_fallback_failure(self) -> bool:
        self._search_fallback_consecutive_failures += 1
        self._stats['search_fallback_consecutive_failures'] = (
            self._search_fallback_consecutive_failures
        )
        if self._search_fallback_consecutive_failures < self._search_fallback_trip_threshold():
            return False

        self._open_search_fallback_circuit()
        return True

    @contextmanager
    def _memory_slot(self):
        if self._is_cancel_requested():
            raise KeyboardInterrupt('Memory operation cancelled by interrupt request.')
        if self._is_pressure_open():
            self._record_pressure_skip()
            raise _MemoryBackpressureError(
                'Memory operation rejected while pressure circuit is open.'
            )

        timeout_seconds = max(0.01, self.config.memory_slot_timeout_ms / 1000.0)
        acquired = self._memory_op_semaphore.acquire(timeout=timeout_seconds)
        if not acquired:
            if self._is_cancel_requested():
                raise KeyboardInterrupt('Memory operation cancelled while waiting for slot.')
            self._record_slot_timeout()
            raise _MemoryBackpressureError('Memory operation slot saturated.')
        try:
            self._record_slot_success()
            yield
        finally:
            self._memory_op_semaphore.release()

    def _is_transient_backend_error(self, error: BaseException) -> bool:
        error_text = str(error).lower()
        return any(marker.lower() in error_text for marker in self.config.transient_error_markers)

    def _run_with_retries(
        self,
        *,
        attempts: int,
        stat_key: str,
        operation: Any,
    ) -> Any:
        def before_sleep(_retry_state: Any) -> None:
            if self._is_cancel_requested():
                raise KeyboardInterrupt('Memory retry loop cancelled by interrupt request.')
            self._stats[stat_key] += 1

        retryer = Retrying(
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

        def run_once() -> Any:
            if self._is_cancel_requested():
                raise KeyboardInterrupt('Memory operation cancelled before execution.')
            with self._memory_slot():
                return operation()

        return retryer(run_once)

    def format_store_memory_text(self, *, question: str, answer: str, reasoning: str) -> str:
        normalized_question = self._normalize_text(question)
        normalized_answer = self._normalize_text(answer)
        normalized_reasoning = self._normalize_text(reasoning)
        parts = [
            f'Answer: {normalized_answer}',
            f'Reasoning: {normalized_reasoning}',
        ]
        if self.config.include_question_in_memory_text:
            parts.insert(0, f'Question: {normalized_question}')
        return self._sanitize_store_content(' | '.join(parts))

    def _parse_legacy_memory_payload(self, passage: str) -> list[str]:
        snippets: list[str] = []
        for line in passage.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except Exception:
                snippets.append(self._normalize_text(candidate))
                continue

            match payload:
                case {
                    'answer': answer,
                    'reasoning': reasoning,
                    **rest,
                }:
                    components = [f'Answer: {self._normalize_text(str(answer))}']
                    components.append(f'Reasoning: {self._normalize_text(str(reasoning))}')
                    if self.config.include_question_in_memory_text and 'question' in rest:
                        components.insert(
                            0,
                            f'Question: {self._normalize_text(str(rest.get("question", "")))}',
                        )
                    snippets.append(' | '.join(components))
                case _:
                    snippets.append(self._normalize_text(candidate))
        return snippets

    def _semantic_recalled_passages(self, passages: list[str]) -> list[str]:
        raw_candidates = [
            self._normalize_text(text) for text in passages if self._normalize_text(text)
        ]
        extracted = (
            list(
                chain.from_iterable(
                    self._parse_legacy_memory_payload(candidate) for candidate in raw_candidates
                )
            )
            if self.config.parse_legacy_json_memory_payloads
            else raw_candidates
        )
        deduplicated = list(dict.fromkeys(extracted))
        limited = deduplicated[: self.config.recall_max_items]
        return self._normalize_passages(limited)

    def render_recalled_memory_text(self, passages: list[str]) -> str:
        cleaned = self._semantic_recalled_passages(passages)
        if not cleaned:
            return 'No previous statements found.'
        if self.config.render_recalled_memory_style == 'plain':
            return '\n'.join(cleaned)
        return '\n'.join(f'- {snippet}' for snippet in cleaned)

    @staticmethod
    def _is_recoverable_memory_error(error: Exception) -> bool:
        error_text = str(error).lower()
        recoverable_markers = (
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
        return any(marker in error_text for marker in recoverable_markers)

    def _run_fallback_search(
        self,
        query: str,
        search_args: dict[str, Any],
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        if self._is_search_fallback_open():
            self._stats['search_fallback_circuit_open_skips'] += 1
            self._stats['search_graceful_empty'] += 1
            skip_count = self._stats['search_fallback_circuit_open_skips']
            if skip_count == 1 or skip_count % self.config.search_fallback_warning_every == 0:
                self._stats['search_fallback_warning_emitted'] += 1
                logger.warning(
                    '[Mem0Tools]: Search fallback suppression active; '
                    f'returning empty recalled memory (skip_count={skip_count}).'
                )
            return Success([])

        self._stats['search_fallback_attempts'] += 1
        fallback_args = {**search_args, 'limit': self.config.search_fallback_limit}
        try:
            result = self._run_with_retries(
                attempts=self._search_fallback_retry_attempts(),
                stat_key='search_transient_retries',
                operation=lambda: self.memory.search(query=query, **fallback_args),
            )
            self._record_search_fallback_success()
            self._stats['search_fallback_success'] += 1
            self._stats['search_success'] += 1
            return self._normalize_search_payload(result)
        except Exception as fallback_error:
            if self.config.enable_resilient_search_fallback:
                self._stats['search_graceful_empty'] += 1
                trip_open = self._record_search_fallback_failure()
                graceful_count = self._stats['search_graceful_empty']
                should_log = (
                    graceful_count == 1
                    or graceful_count % self.config.search_fallback_warning_every == 0
                    or trip_open
                )
                if should_log:
                    self._stats['search_fallback_warning_emitted'] += 1
                    if trip_open:
                        logger.warning(
                            '[Mem0Tools]: Recoverable search fallback repeatedly failed; '
                            'opening fallback suppression circuit and continuing with empty '
                            f'recalled memory (count={graceful_count}).'
                        )
                    else:
                        logger.warning(
                            '[Mem0Tools]: Recoverable search fallback failed; '
                            f'continuing with empty recalled memory (count={graceful_count}).'
                        )
                return Success([])

            return Failure(
                MemorySearchError(
                    message='Mem0 search fallback failed.',
                    cause=fallback_error,
                )
            )

    def _supports_dimension_fallback(self) -> bool:
        embedding_dims = None
        if self.config.vector_store is not None:
            embedding_dims = self.config.vector_store.config.embedding_model_dims
        return bool(self.config.enable_dimension_fallback and embedding_dims)

    @staticmethod
    def _is_dimension_related_error(error: Exception) -> bool:
        error_text = str(error).lower()
        dimension_markers = (
            'dimension',
            'dimensions',
            'matryoshka',
            'embedding_model_dims',
            'size mismatch',
            'expected dim',
        )
        return any(marker in error_text for marker in dimension_markers)

    def _attempt_dimension_fallback_reinit(self, error: Exception) -> bool:
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
            self.memory = self._initialize_provider(strip_embedding_dims=True)
            self._using_dimensionless_config = True
            self._stats['dimension_fallback_activations'] += 1
        except MemoryConfigurationError as reinit_error:
            logger.error(f'[Mem0Tools]: Dimension fallback initialization failed: {reinit_error}')
            return False
        else:
            return True

    def stats_snapshot(self) -> dict[str, int]:
        return {
            **self._stats,
            'using_dimensionless_config': int(self._using_dimensionless_config),
        }

    def _store_add(
        self,
        *,
        content: str,
        user_id: str | None,
        metadata: dict[str, Any],
        infer: bool,
    ) -> None:
        _ = self._run_with_retries(
            attempts=self.config.store_retry_attempts,
            stat_key='store_transient_retries',
            operation=lambda: self.memory.add(
                content,
                user_id=user_id,
                metadata=metadata,
                infer=infer,
            ),
        )

    def _store_failure(self, error: Exception) -> Result[None, MemoryToolError]:
        self._stats['store_failures'] += 1
        return Failure(
            MemoryStoreError(
                message='Mem0 add operation failed while storing memory.',
                cause=error,
            )
        )

    def _store_with_dimension_fallback(
        self,
        *,
        normalized_content: str,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> Result[None, MemoryToolError]:
        try:
            self._store_add(
                content=normalized_content,
                user_id=user_id,
                metadata=metadata,
                infer=self.config.memory_add_infer,
            )
            self._stats['store_success'] += 1
            self._stats['dimension_fallback_retry_success'] += 1
            return Success(None)
        except Exception as retry_error:
            if self._is_recoverable_memory_error(retry_error):
                self._stats['store_recoverable_failures'] += 1
                return Success(None)
            return self._store_failure(retry_error)

    def _store_with_infer_false_retry(
        self,
        *,
        normalized_content: str,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> Result[None, MemoryToolError]:
        self._stats['store_infer_false_retries'] += 1
        try:
            self._store_add(
                content=normalized_content,
                user_id=user_id,
                metadata=metadata,
                infer=False,
            )
            self._stats['store_infer_false_retry_success'] += 1
            self._stats['store_success'] += 1
            return Success(None)
        except Exception as inferless_error:
            if self._is_recoverable_memory_error(inferless_error):
                self._stats['store_recoverable_failures'] += 1
                return Success(None)
            return self._store_failure(inferless_error)

    def _handle_store_error(
        self,
        *,
        error: Exception,
        normalized_content: str,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> Result[None, MemoryToolError]:
        if self._attempt_dimension_fallback_reinit(error):
            return self._store_with_dimension_fallback(
                normalized_content=normalized_content,
                user_id=user_id,
                metadata=metadata,
            )

        if not self._is_recoverable_memory_error(error):
            return self._store_failure(error)

        if self.config.memory_add_infer:
            return self._store_with_infer_false_retry(
                normalized_content=normalized_content,
                user_id=user_id,
                metadata=metadata,
            )

        self._stats['store_recoverable_failures'] += 1
        return Success(None)

    def store_memory(
        self,
        content: str | list[dict[str, Any]],
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result[None, MemoryToolError]:
        """Persist memory entries for a user scope."""
        if self._is_cancel_requested():
            raise KeyboardInterrupt('Memory store cancelled by interrupt request.')
        self._stats['store_attempts'] += 1
        if self.config.memory_pressure_mode in {'disabled', 'read_only'}:
            self._stats['store_skipped_pressure_mode'] += 1
            return Success(None)

        raw_content = content if isinstance(content, str) else str(content)
        normalized_content = self._sanitize_store_content(raw_content)
        if not normalized_content:
            self._stats['store_skipped_empty_content'] += 1
            return Success(None)

        try:
            self._store_add(
                content=normalized_content,
                user_id=user_id,
                metadata=metadata or {},
                infer=self.config.memory_add_infer,
            )
            self._stats['store_success'] += 1
            return Success(None)
        except _MemoryBackpressureError:
            if self.config.drop_store_on_backpressure:
                self._stats['store_backpressure_skips'] += 1
                self._pressure_warn('store')
                return Success(None)
            return self._store_failure(
                _MemoryBackpressureError('Mem0 store backpressure saturation.')
            )
        except Exception as error:
            return self._handle_store_error(
                error=error,
                normalized_content=normalized_content,
                user_id=user_id,
                metadata=metadata or {},
            )

    def _search_once(
        self,
        query: str,
        search_args: dict[str, Any],
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        if self._is_cancel_requested():
            raise KeyboardInterrupt('Memory search cancelled by interrupt request.')
        self._stats['search_attempts'] += 1
        normalized_query = self._sanitize_query(query)
        if not normalized_query:
            self._stats['search_graceful_empty'] += 1
            return Success([])
        try:
            result = self._run_with_retries(
                attempts=self.config.search_retry_attempts,
                stat_key='search_transient_retries',
                operation=lambda: self.memory.search(query=normalized_query, **search_args),
            )
        except _MemoryBackpressureError:
            return self._search_on_backpressure()
        except Exception as error:
            if self._attempt_dimension_fallback_reinit(error):
                return self._search_after_dimension_fallback(normalized_query, search_args)

            if self._is_recoverable_memory_error(error):
                self._stats['search_recoverable_failures'] += 1
                return self._run_fallback_search(normalized_query, search_args)

            self._stats['search_failures'] += 1
            return Failure(
                MemorySearchError(
                    message='Mem0 search operation failed.',
                    cause=error,
                )
            )
        self._stats['search_success'] += 1
        return self._normalize_search_payload(result)

    def _search_on_backpressure(self) -> Result[list[MemoryRecord], MemoryToolError]:
        if self.config.degrade_search_on_backpressure:
            self._stats['search_backpressure_degrades'] += 1
            self._stats['search_graceful_empty'] += 1
            self._pressure_warn('search')
            return Success([])

        self._stats['search_failures'] += 1
        return Failure(
            MemorySearchError(
                message='Mem0 search blocked by pressure backoff policy.',
                cause=_MemoryBackpressureError('Mem0 search backpressure saturation.'),
            )
        )

    def _search_after_dimension_fallback(
        self,
        normalized_query: str,
        search_args: dict[str, Any],
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        try:
            result = self._run_with_retries(
                attempts=self.config.search_retry_attempts,
                stat_key='search_transient_retries',
                operation=lambda: self.memory.search(query=normalized_query, **search_args),
            )
            self._stats['search_success'] += 1
            self._stats['dimension_fallback_retry_success'] += 1
            return self._normalize_search_payload(result)
        except Exception as retry_error:
            if self._is_recoverable_memory_error(retry_error):
                self._stats['search_recoverable_failures'] += 1
                return self._run_fallback_search(normalized_query, search_args)
            self._stats['search_failures'] += 1
            return Failure(
                MemorySearchError(
                    message='Mem0 search operation failed.',
                    cause=retry_error,
                )
            )

    @staticmethod
    def _build_search_args(
        user_id: str | None,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {'limit': limit}
        if filters:
            args['filters'] = filters
        if user_id:
            args['user_id'] = user_id
        return args

    @staticmethod
    def _normalize_search_payload(
        payload: Any,
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        if isinstance(payload, list):
            normalized_list: list[MemoryRecord] = [cast(MemoryRecord, item) for item in payload]
            return Success(normalized_list)

        if isinstance(payload, dict):
            raw_results = payload.get('results')
            if isinstance(raw_results, list):
                normalized_items: list[MemoryRecord] = [
                    cast(MemoryRecord, item) for item in raw_results if isinstance(item, dict)
                ]
                return Success(normalized_items)

            return Failure(
                MemoryContractError(
                    message='Mem0 search payload is missing list-like `results`.',
                )
            )

        return Failure(
            MemoryContractError(
                message='Mem0 search returned unsupported payload type.',
            )
        )

    @staticmethod
    def _extract_passages(items: list[MemoryRecord]) -> list[str]:
        return [
            text
            for item in items
            for text in [item.get('memory') or item.get('text') or '']
            if text
        ]

    def _normalize_passages(self, passages: list[str]) -> list[str]:
        normalized_passages: list[str] = []
        total_chars = 0
        for passage in passages:
            normalized = self._truncate(
                self._normalize_text(passage), self.config.max_passage_chars
            )
            if not normalized:
                continue
            remaining = self.config.max_recalled_chars - total_chars
            if remaining <= 0:
                return normalized_passages
            clipped = normalized[:remaining].rstrip()
            if not clipped:
                return normalized_passages
            normalized_passages.append(clipped)
            total_chars += len(clipped)
        return normalized_passages

    @staticmethod
    def _extract_memory_ids(payload: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
        match payload:
            case {'results': list(items)}:
                return [
                    mem_id
                    for item in items
                    if isinstance(item, dict)
                    for mem_id in [item.get('id')]
                    if isinstance(mem_id, str) and mem_id
                ]
            case _:
                return []

    def search_memories(
        self,
        query: str | list[str],
        user_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Result[MemorySearchResult, MemoryToolError]:
        """Search memories and return normalized payload with passages list."""
        if self._is_cancel_requested():
            raise KeyboardInterrupt('Memory search cancelled by interrupt request.')
        queries = [query] if isinstance(query, str) else query
        resolved_limit = limit or self.config.recall_top_k
        search_args = self._build_search_args(
            user_id=user_id, limit=resolved_limit, filters=filters
        )

        all_items: list[MemoryRecord] = []
        for one_query in queries:
            result = self._search_once(one_query, search_args)
            if isinstance(result, Failure):
                if self.config.enable_resilient_search_fallback:
                    self._stats['search_graceful_empty'] += 1
                    return Success({'passages': [], 'count': 0})
                return result
            all_items.extend(result.unwrap())

        passages = self._semantic_recalled_passages(self._extract_passages(all_items))
        return Success({
            'passages': passages,
            'count': len(passages),
        })

    def clear_user_memory(self, user_id: str) -> Result[int, MemoryToolError]:
        """Clear all memory entries for a given user id."""
        self._stats['clear_attempts'] += 1
        try:
            listed = self.memory.get_all(user_id=user_id, limit=10_000)
            memory_ids = self._extract_memory_ids(listed)

            for mem_id in memory_ids:
                self.memory.delete(mem_id)

            deleted = len(memory_ids)
            logger.info(f'Cleared {deleted} memories for {user_id}')
            self._stats['clear_success'] += 1
            return Success(deleted)
        except Exception as error:
            logger.error(f'[Mem0Tools]: Mem0 clear failed: {error}')
            self._stats['clear_failures'] += 1
            return Failure(
                MemoryClearError(
                    message='Mem0 clear operation failed.',
                    cause=error,
                )
            )
