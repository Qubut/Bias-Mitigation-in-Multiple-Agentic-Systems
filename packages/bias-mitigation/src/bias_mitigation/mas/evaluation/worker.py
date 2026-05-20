from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dspy

from .models import _AgentTurnRow, _EvaluationMetadata, _RoundMetricRow, _SampleOutcomeRow


@dataclass(slots=True)
class _ExampleEvalRecord:
    """Per-example evaluation values plus stratification metadata.

    ``y_true`` is the gold label index (``inputs['label']``) and ``y_pred``
    is the index of the consensus answer chosen by majority vote over each
    agent's final answer. Both are ``None`` when the gold answer or the
    final answers cannot be resolved (e.g. malformed example, agent crash);
    rows with missing values are excluded from Fairlearn fairness
    aggregations.
    """

    metrics: dict[str, float]
    metadata: _EvaluationMetadata
    sample_outcome: _SampleOutcomeRow
    agent_turns: list[_AgentTurnRow]
    round_metric_rows: list[_RoundMetricRow]
    y_true: int | None = None
    y_pred: int | None = None


@dataclass(frozen=True, slots=True)
class _ParallelEvalTask:
    """Immutable task payload carried alongside each evaluated example.

    Built up-front by :meth:`MASEvaluator._build_parallel_tasks` so the
    ``dspy.Evaluate`` metric callback can recover the stable sample id
    and pre-computed metadata for the example it receives.
    """

    example_index: int
    example: dspy.Example
    inputs: dict[str, Any]
    metadata: _EvaluationMetadata
    sample_id: str


@dataclass(frozen=True, slots=True)
class _ParallelEvalResult:
    """Envelope passed to stream sinks for a single evaluated example."""

    task: _ParallelEvalTask
    record: _ExampleEvalRecord | None
    error: str | None
