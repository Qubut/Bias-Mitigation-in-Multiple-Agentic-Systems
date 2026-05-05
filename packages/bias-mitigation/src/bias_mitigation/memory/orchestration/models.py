"""Typed contracts for memory orchestration runtime."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryOrchestrationConfig:
    """State-machine and worker settings for Mem0 orchestration."""

    worker_threads: int = 4
    max_pending_store_tasks: int = 128
    recall_timeout_ms: int = 6000
    store_timeout_ms: int = 2500
    store_async: bool = True
    failure_trip_threshold: int = 8
    recovery_success_threshold: int = 6


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Normalized recall response consumed by agents."""

    text: str
    count: int
    status: str
