"""Typed contracts for memory tooling interfaces and payloads."""

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict


class MemoryRecord(TypedDict, total=False):
    """One memory row returned by provider search/get APIs."""

    id: str
    memory: str
    text: str


class MemoryResultEnvelope(TypedDict):
    """Provider envelope containing memory result rows."""

    results: list[MemoryRecord]


class MemorySearchResult(TypedDict):
    """Normalized search output returned by Mem0Tools."""

    passages: list[str]
    count: int


class MemoryProviderProtocol(Protocol):
    """Protocol expected from backing memory provider implementation."""

    def add(
        self,
        content: str | list[dict[str, Any]],
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def search(self, query: str, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]: ...

    def get_all(
        self, user_id: str, limit: int = 10_000
    ) -> dict[str, Any] | list[dict[str, Any]]: ...

    def delete(self, memory_id: str) -> Any: ...
