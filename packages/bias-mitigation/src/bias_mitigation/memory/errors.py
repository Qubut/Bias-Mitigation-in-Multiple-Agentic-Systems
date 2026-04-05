"""Custom error hierarchy for memory tooling operations."""

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryToolError(Exception):
    """Base memory-tool exception with optional causal chain."""

    message: str
    cause: Exception | None = None

    def __str__(self) -> str:
        if self.cause is None:
            return self.message
        return f'{self.message} (cause={self.cause})'


@dataclass(slots=True)
class MemoryConfigurationError(MemoryToolError):
    """Configuration or initialization failure for memory providers."""


@dataclass(slots=True)
class MemoryStoreError(MemoryToolError):
    """Failure while storing memory entries."""


@dataclass(slots=True)
class MemorySearchError(MemoryToolError):
    """Failure while searching memory entries."""


@dataclass(slots=True)
class MemoryClearError(MemoryToolError):
    """Failure while clearing memory entries."""


@dataclass(slots=True)
class MemoryContractError(MemoryToolError):
    """Unexpected provider payload violating expected schema contract."""
