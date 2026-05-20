"""Data contracts shared by the evaluation and training workflow runtimes.

The workflow layer is intentionally split into a stateless ``WorkflowRuntime``
protocol (the pipeline stages) and the data objects defined here. Stages read
from and mutate a single ``RunContext`` instance so that orchestration,
persistence, and side effects (MLflow logging, mem0 calls, GEPA compilation)
remain testable in isolation.

Two dataclasses live here:

* ``RunRequest`` is the immutable user-supplied envelope (CLI flags,
  dataset paths, MLflow tracking URI, optional GEPA / resume settings).
* ``RunContext`` is the mutable container that pipeline stages progressively
  fill in (loaded configs, dataset splits, the built DSPy program, the
  evaluator, results, error state).

``RunMode`` distinguishes EVALUATE-only runs from full TRAIN runs that also
invoke GEPA prompt optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import dspy
from mlflow.entities import Run

from bias_mitigation.data.models.config import MASConfig


class RunMode(StrEnum):
    """Discriminator that selects which workflow runtime executes a request.

    ``EVALUATE`` runs the MAS against the dev split and logs fairness /
    accuracy metrics to MLflow. ``TRAIN`` additionally compiles the program
    with GEPA on the train split and writes the optimized DSPy module to
    disk before evaluating on a held-out slice.

    Attributes:
        EVALUATE: Evaluation-only mode; no prompt optimization is performed.
        TRAIN: Full training mode that runs GEPA followed by hold-out
            validation on the optimized program.
    """

    EVALUATE = 'evaluate'
    TRAIN = 'train'


@dataclass(frozen=True, slots=True)
class RunRequest:
    """User-facing description of a single evaluation or training run.

    A ``RunRequest`` is the only object a CLI entrypoint constructs before
    handing control to the workflow state machine. It is intentionally frozen
    so downstream stages can rely on its values not changing mid-run, and
    so it can be safely fingerprinted for resume / reproducibility purposes.

    Attributes:
        mode: Whether to run a pure evaluation or a full GEPA training cycle.
        config_path: Path to the YAML MAS configuration (``MASConfig``).
        dataset_dir: Directory containing the versioned dataset splits
            (train / dev) tracked by ``dataset_tracker``.
        tracking_uri: MLflow tracking server URI (file:// or http://).
        run_name: Human-readable label attached to the MLflow run.
        intervention: Optional override for the bias-mitigation strategy
            (e.g. ``mem0``, ``mem0g``, ``mem0g_gepa``); ``None`` keeps the
            value from the YAML config.
        memory_config: Optional secondary YAML merged into ``MASConfig`` to
            inject memory backend credentials/parameters.
        min_system_robustness: Optional MLflow validation threshold; if the
            ``MAS_System_Robustness`` metric falls below this, the run is
            marked as failed validation.
        run_safety_scan: When ``True`` triggers the optional Giskard-based
            safety scan during the persist stage.
        subset: Cap on the number of dev examples used for evaluation.
            ``-1`` (or ``<= 0``) means use the full dev split.
        subset_seed: Seed for stratified subset selection; pinning this
            guarantees reproducible example selection across runs.
        optimized_program_path: Path to a previously-saved DSPy program to
            reload before evaluation (used to evaluate a GEPA-optimized
            program without retraining).
        skip_validation: Training-mode flag that skips the post-GEPA
            hold-out validation step.
        train_subset: Cap on the GEPA trainset size. ``-1`` keeps the full
            train split; this is independent from ``subset`` (eval cap).
        valset_size_override: CLI-level override for
            ``GepaConfig.valset_size``; ``None`` keeps the YAML value.
        resume_from: Path to a ``stream_metric_rows.jsonl`` from a prior
            interrupted run. Set to resume that run instead of starting fresh.
    """

    mode: RunMode
    config_path: Path
    dataset_dir: Path
    tracking_uri: str
    run_name: str
    intervention: str | None
    memory_config: Path | None
    min_system_robustness: float | None
    run_safety_scan: bool
    subset: int
    subset_seed: int
    # Training-mode fields (None / defaults when mode=EVALUATE)
    optimized_program_path: Path | None = None
    skip_validation: bool = False
    train_subset: int = -1  # cap on GEPA trainset (-1 = full); separate from subset (eval cap)
    valset_size_override: int | None = None  # override GepaConfig.valset_size from CLI
    # Resume-mode field: path to a stream_metric_rows.jsonl from a prior interrupted run
    resume_from: Path | None = None


@dataclass(slots=True)
class RunContext:
    """Mutable carrier threaded through the workflow pipeline stages.

    Each stage (``prepare``, ``build``, ``execute``, ``persist``, ``fail``)
    reads from the context, mutates it in place, and returns it. This keeps
    stage signatures uniform and makes it trivial to inspect a run's full
    state at any point (helpful for debugging GEPA / mem0 interactions).

    Fields begin life either at their dataclass defaults or set from the
    incoming ``RunRequest``; later stages enrich the object with the loaded
    configuration, dataset examples, the built DSPy program, the evaluator,
    raw evaluation outputs, and persistence side-effect tracking.

    Attributes:
        request: The immutable run request this context is bound to.
        started_at_utc: ISO-8601 UTC timestamp captured at context creation
            (used for live-run directory naming and MLflow tags).
        mas_config: Resolved ``MASConfig`` (merged YAML + CLI overrides);
            populated by ``prepare``.
        container: ``dependency-injector`` ``Container`` used to construct
            the MAS program and its memory tools; populated by ``prepare``.
        train_examples: Train-split DSPy examples (capped by
            ``train_subset`` in TRAIN mode).
        dev_examples: Dev-split DSPy examples loaded from the dataset
            directory.
        gepa_val_examples: Non-overlapping subset of ``dev_examples`` used
            by the GEPA Pareto tracker. Never used for hold-out evaluation.
        holdout_examples: Disjoint subset of ``dev_examples`` reserved for
            post-GEPA hold-out validation; GEPA never observes these.
        evaluated_examples: Stratified subset of ``dev_examples`` actually
            evaluated by the runtime (after any resume filtering).
        train_ds_input: MLflow dataset input handle for the train split,
            logged as ``context='training'``.
        dev_ds_input: MLflow dataset input handle for the dev split, logged
            as ``context='evaluation'``.
        effective_subset: Final number of examples in the evaluation subset
            (before resume filtering).
        example_index_offset: When resuming, the count of already-completed
            examples; downstream streaming uses this so global indices match
            the original run.
        run_metadata: Flat string/string dict of run-identifying metadata
            (intervention, protocol, models, seeds) mirrored into MLflow
            tags and local stream manifests.
        active_run: The MLflow ``Run`` object started by ``build``.
        mas_program: The compiled DSPy multi-agent module under evaluation.
        evaluator: The ``MASEvaluator`` instance built in ``build``.
        result: Raw dictionary returned by ``MASEvaluator`` (metrics, sample
            outcomes, agent turns, stream rows, etc.).
        memory_stats: Snapshot of mem0 tool counters captured after
            ``execute`` (read/write counts, cache hits, etc.).
        safety_scan_status: Outcome of the optional Giskard safety scan
            triggered when ``request.run_safety_scan`` is ``True``.
        error: Free-form error message recorded when the state machine
            transitions into the ``failed`` state.
        optimized_program: TRAIN-mode artefact: the DSPy program returned
            by GEPA compilation.
        gepa_stats: TRAIN-mode statistics (Pareto frontier, iterations,
            best score, wall time) emitted by the GEPA optimizer.
        optimized_program_path: TRAIN-mode artefact: filesystem path to the
            serialized optimized program.
    """

    request: RunRequest
    started_at_utc: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    mas_config: MASConfig | None = None
    container: Any | None = None
    train_examples: list[dspy.Example] = field(default_factory=list)
    dev_examples: list[dspy.Example] = field(default_factory=list)
    # TRAIN mode: non-overlapping 3-way split of dev_examples.
    # gepa_val_examples → GEPA Pareto tracker (never used for hold-out eval).
    # holdout_examples  → post-GEPA validation (GEPA never sees these).
    # In EVALUATE mode both are empty; dev_examples is used directly.
    gepa_val_examples: list[dspy.Example] = field(default_factory=list)
    holdout_examples: list[dspy.Example] = field(default_factory=list)
    evaluated_examples: list[dspy.Example] = field(default_factory=list)
    train_ds_input: Any | None = None
    dev_ds_input: Any | None = None
    effective_subset: int = 0
    example_index_offset: int = 0  # number of already-completed examples when resuming
    run_metadata: dict[str, str] = field(default_factory=dict)
    active_run: Run | None = None
    mas_program: dspy.Module | None = None
    evaluator: Any | None = None
    result: dict[str, Any] = field(default_factory=dict)
    memory_stats: dict[str, int] = field(default_factory=dict)
    safety_scan_status: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    # Training-mode artefacts (populated only when mode=TRAIN)
    optimized_program: dspy.Module | None = None
    gepa_stats: dict[str, Any] = field(default_factory=dict)
    optimized_program_path: Path | None = None
