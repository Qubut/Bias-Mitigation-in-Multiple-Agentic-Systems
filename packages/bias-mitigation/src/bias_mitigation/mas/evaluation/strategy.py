"""Backend strategy abstractions for evaluator orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import dspy


class EvaluationStrategy(Protocol):
    """Common interface for evaluator backend strategies."""

    def evaluate(
        self,
        *,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute backend-specific evaluation."""


@dataclass(slots=True)
class DeterministicStrategy(EvaluationStrategy):
    """Strategy that delegates to deterministic evaluator execution."""

    execute: Callable[..., dict[str, Any]]

    def evaluate(
        self,
        *,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self.execute(
            program=program,
            devset=devset,
            extra_metadata=extra_metadata,
        )


@dataclass(slots=True)
class GenAIStrategy(EvaluationStrategy):
    """Strategy that delegates to GenAI evaluator execution."""

    execute: Callable[..., dict[str, Any]]

    def evaluate(
        self,
        *,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self.execute(
            program=program,
            devset=devset,
            extra_metadata=extra_metadata,
        )


class EvaluatorStrategyFactory:
    """Factory for evaluator backend strategy selection."""

    @staticmethod
    def create(
        *,
        backend: str,
        deterministic_execute: Callable[..., dict[str, Any]],
        genai_execute: Callable[..., dict[str, Any]],
    ) -> EvaluationStrategy:
        if backend == 'genai':
            return GenAIStrategy(execute=genai_execute)
        return DeterministicStrategy(execute=deterministic_execute)
