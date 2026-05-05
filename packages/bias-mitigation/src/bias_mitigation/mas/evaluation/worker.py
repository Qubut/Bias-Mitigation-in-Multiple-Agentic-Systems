from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dspy
from returns.pipeline import pipe
from returns.pointfree import lash, map_
from returns.result import Success, safe

from .models import _AgentTurnRow, _EvaluationMetadata, _RoundMetricRow, _SampleOutcomeRow

if TYPE_CHECKING:
    from ..evaluator import MASEvaluator
    from .adapters import PredictFnAdapter


@dataclass(slots=True)
class _ExampleEvalRecord:
    """Per-example evaluation values plus stratification metadata."""

    metrics: dict[str, float]
    metadata: _EvaluationMetadata
    sample_outcome: _SampleOutcomeRow
    agent_turns: list[_AgentTurnRow]
    round_metric_rows: list[_RoundMetricRow]


@dataclass(frozen=True, slots=True)
class _ParallelEvalTask:
    """Immutable task payload for deterministic parallel evaluation."""

    example_index: int
    example: dspy.Example
    inputs: dict[str, Any]
    metadata: _EvaluationMetadata
    sample_id: str


@dataclass(frozen=True, slots=True)
class _ParallelEvalResult:
    """Worker result envelope used by ordered reduction."""

    task: _ParallelEvalTask
    record: _ExampleEvalRecord | None
    error: str | None


class _DeterministicParallelWorker(dspy.Module):
    """Per-sample deterministic worker invoked via ``dspy.Parallel``."""

    def __init__(
        self,
        *,
        predict_fn: PredictFnAdapter,
        evaluator: MASEvaluator,
        extra_metadata: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        self._predict_fn = predict_fn
        self._evaluator = evaluator
        self._extra_metadata = extra_metadata

    def forward(self, task: _ParallelEvalTask) -> _ParallelEvalResult:
        if self._evaluator.is_cancel_requested():
            return _ParallelEvalResult(
                task=task,
                record=None,
                error='Evaluation cancelled by interrupt request.',
            )
        return pipe(
            safe(self._evaluator.evaluate_single_example)(
                predict_fn=self._predict_fn,
                example=task.example,
                extra_metadata=self._extra_metadata,
                example_index=task.example_index,
            ),
            map_(lambda record: _ParallelEvalResult(task=task, record=record, error=None)),
            lash(lambda err: Success(_ParallelEvalResult(task=task, record=None, error=str(err)))),
        ).unwrap()
