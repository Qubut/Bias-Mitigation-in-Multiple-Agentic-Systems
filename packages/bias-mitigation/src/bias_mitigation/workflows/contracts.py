"""Shared workflow contracts for declarative run orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import dspy
from mlflow.entities import Run

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.evaluator import EvaluatorBackend


class RunMode(StrEnum):
    """Operational mode for shared workflow orchestration."""

    EVALUATE = 'evaluate'
    TRAIN = 'train'


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Immutable request envelope used by workflow state machines."""

    mode: RunMode
    config_path: Path
    dataset_dir: Path
    tracking_uri: str
    run_name: str
    intervention: str | None
    memory_config: Path | None
    evaluator_backend: EvaluatorBackend
    min_system_robustness: float | None
    run_safety_scan: bool
    subset: int
    subset_seed: int


@dataclass(slots=True)
class RunContext:
    """Mutable run context updated by declarative workflow states."""

    request: RunRequest
    started_at_utc: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    mas_config: MASConfig | None = None
    container: Any | None = None
    train_examples: list[dspy.Example] = field(default_factory=list)
    dev_examples: list[dspy.Example] = field(default_factory=list)
    evaluated_examples: list[dspy.Example] = field(default_factory=list)
    train_ds_input: Any | None = None
    dev_ds_input: Any | None = None
    effective_subset: int = 0
    run_metadata: dict[str, str] = field(default_factory=dict)
    active_run: Run | None = None
    mas_program: dspy.Module | None = None
    evaluator: Any | None = None
    result: dict[str, Any] = field(default_factory=dict)
    memory_stats: dict[str, int] = field(default_factory=dict)
    safety_scan_status: dict[str, str] = field(default_factory=dict)
    error: str | None = None
