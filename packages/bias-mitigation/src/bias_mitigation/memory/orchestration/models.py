"""Typed contracts for memory orchestration runtime.

The Pydantic :class:`bias_mitigation.data.models.config.MemoryOrchestrationConfig`
is the single source of truth for orchestrator tuning knobs; the runtime
:class:`MemoryOrchestrator` accepts it directly so there is no second
shadow dataclass to keep in sync.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Normalized recall response consumed by agents."""

    text: str
    count: int
    status: str
