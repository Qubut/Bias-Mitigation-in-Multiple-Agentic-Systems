"""Declarative Mem0 tooling wrapper using typed Result flows."""

from itertools import chain
from typing import Any, cast

from loguru import logger
from mem0 import Memory
from returns.iterables import Fold
from returns.pipeline import flow
from returns.result import Failure, Result, Success

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


class Mem0Tools:
    """Adapter/repository abstraction around Mem0 operations.

    Architecture:
    - Pure helpers: argument building + payload normalization + projection.
    - Effectful gateways: provider `add/search/get_all/delete` calls only.
    """

    def __init__(self, config: Mem0Config):
        self.config = config
        logger.info('[Mem0Tools]: Initializing Mem0')
        try:
            self.memory: MemoryProviderProtocol = Memory.from_config(self.config.to_mem0_dict())
        except Exception as error:
            raise MemoryConfigurationError(
                message='Failed to initialize Mem0 provider from configuration.',
                cause=error,
            ) from error

    def store_memory(
        self,
        content: str | list[dict[str, Any]],
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result[None, MemoryToolError]:
        """Persist memory entries for a user scope."""
        try:
            self.memory.add(content, user_id=user_id, metadata=metadata or {})
        except Exception as error:
            return Failure(
                MemoryStoreError(
                    message='Mem0 add operation failed while storing memory.',
                    cause=error,
                )
            )
        return Success(None)

    def _search_once(
        self,
        query: str,
        search_args: dict[str, Any],
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        try:
            result = self.memory.search(query=query, **search_args)
        except Exception as error:
            return Failure(
                MemorySearchError(
                    message='Mem0 search operation failed.',
                    cause=error,
                )
            )

        return self._normalize_search_payload(result)

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
        payload: dict[str, Any] | list[dict[str, Any]],
    ) -> Result[list[MemoryRecord], MemoryToolError]:
        match payload:
            case {'results': list(raw_results)}:
                normalized_items: list[MemoryRecord] = [
                    cast(MemoryRecord, item) for item in raw_results if isinstance(item, dict)
                ]
                return Success(normalized_items)
            case list(raw_results):
                normalized_list: list[MemoryRecord] = [
                    cast(MemoryRecord, item) for item in raw_results
                ]
                return Success(normalized_list)
            case {'results': _}:
                return Failure(
                    MemoryContractError(
                        message='Mem0 search payload is missing list-like `results`.',
                    )
                )
            case _:
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
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> Result[MemorySearchResult, MemoryToolError]:
        """Search memories and return normalized payload with passages list."""
        queries = [query] if isinstance(query, str) else query
        search_args = self._build_search_args(user_id=user_id, limit=limit, filters=filters)

        query_results = [self._search_once(one_query, search_args) for one_query in queries]
        collected_results = Fold.collect(query_results, Success([]))

        return flow(
            collected_results,
            lambda result: result.map(lambda items: list(chain.from_iterable(items))),
            lambda result: result.map(self._extract_passages),
            lambda result: result.map(
                lambda passages: {
                    'passages': passages,
                    'count': len(passages),
                }
            ),
        )

    def clear_user_memory(self, user_id: str) -> Result[int, MemoryToolError]:
        """Clear all memory entries for a given user id."""
        try:
            listed = self.memory.get_all(user_id=user_id, limit=10_000)
            memory_ids = self._extract_memory_ids(listed)

            for mem_id in memory_ids:
                self.memory.delete(mem_id)

            deleted = len(memory_ids)
            logger.info(f'Cleared {deleted} memories for {user_id}')
            return Success(deleted)
        except Exception as error:
            logger.error(f'[Mem0Tools]: Mem0 clear failed: {error}')
            return Failure(
                MemoryClearError(
                    message='Mem0 clear operation failed.',
                    cause=error,
                )
            )
