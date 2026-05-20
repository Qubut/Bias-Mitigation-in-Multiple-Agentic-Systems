"""I/O Adapter, run-level Specification, and pair-intersection logic.

Responsibilities:
    * Parse ``runs_index.jsonl`` into :class:`RunManifest` value objects with
      strict schema validation.
    * Filter the universe of runs through a composable :class:`RunSpec`
      (Specification pattern with ``&`` / ``|`` / ``~`` combinators).
    * Load per-run ``stream_metric_rows.csv`` into a typed Polars dataframe.
    * Build the 4-way prompt-intersection pair table used by the paired
      analysis notebooks, with explicit pair-consistency checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from operator import and_, or_
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from bias_mitigation.analysis.config import ANALYSIS_CONFIG

__all__ = [
    'INTERVENTION_DISPLAY_LABELS',
    'METRIC_COLUMNS',
    'RunManifest',
    'RunSpec',
    'intersect_pair_table',
    'load_metric_rows',
    'load_round_metrics',
    'load_runs_index',
    'resolve_paper_arm',
    'select_runs',
    'verify_pair_consistency',
]


METRIC_COLUMNS: Final[tuple[str, ...]] = ANALYSIS_CONFIG.metric_columns
INTERVENTION_DISPLAY_LABELS: Final[Mapping[str, str]] = ANALYSIS_CONFIG.intervention_display_labels


class RunManifest(BaseModel):
    """Frozen Pydantic value object representing one row of ``runs_index.jsonl``.

    ``model_config = ConfigDict(extra='forbid', frozen=True)`` makes schema
    drift in the upstream pipeline surface as a validation error at load
    time rather than 200 lines later in an analysis cell.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    run_id: str
    run_dir_name: str
    run_dir_path: str
    run_name: str
    experiment_id: str
    experiment_name: str
    backend: str
    intervention: str
    protocol: str
    agent_model_map: Mapping[str, str]
    subset: int = Field(ge=0)
    subset_seed: int
    started_at_utc: str


def load_runs_index(path: Path) -> tuple[RunManifest, ...]:
    """Load and validate every line of a ``runs_index.jsonl`` file.

    Args:
        path: Absolute path to the ``runs_index.jsonl`` file.

    Returns:
        Validated, immutable tuple of :class:`RunManifest`. Empty if the
        file does not exist or contains no parsable rows.

    Raises:
        pydantic.ValidationError: If any row violates the manifest schema.
    """
    if not path.exists():
        return ()
    with path.open(encoding='utf-8') as handle:
        return tuple(
            RunManifest.model_validate_json(line)
            for line in (raw.strip() for raw in handle)
            if line
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Composable predicate over :class:`RunManifest` (Specification pattern).

    Combine with ``&`` (logical AND), ``|`` (logical OR), and ``~``
    (logical NOT) to build inclusion criteria declaratively without
    branching at every analysis call site.

    Example:
        >>> spec = (
        ...     RunSpec.run_name_eq('MAS_Experiment_Cooperative')
        ...     & RunSpec.intervention_in({'baseline', 'mem0g'})
        ...     & RunSpec.dir_must_exist(Path('/runs'))
        ... )
        >>> # spec(manifest) -> bool

    Attributes:
        predicate: Callable that returns ``True`` if a manifest is admitted.
        label: Human-readable description; combinators concatenate labels.
    """

    predicate: Callable[[RunManifest], bool]
    label: str = ''

    def __call__(self, manifest: RunManifest) -> bool:
        return self.predicate(manifest)

    def __and__(self, other: RunSpec) -> RunSpec:
        return RunSpec(
            predicate=lambda m: self.predicate(m) and other.predicate(m),
            label=f'({self.label} & {other.label})',
        )

    def __or__(self, other: RunSpec) -> RunSpec:
        return RunSpec(
            predicate=lambda m: self.predicate(m) or other.predicate(m),
            label=f'({self.label} | {other.label})',
        )

    def __invert__(self) -> RunSpec:
        return RunSpec(
            predicate=lambda m: not self.predicate(m),
            label=f'~{self.label}',
        )

    @classmethod
    def run_name_eq(cls, name: str) -> RunSpec:
        """Spec admitting manifests whose ``run_name`` equals ``name``."""
        return cls(predicate=lambda m: m.run_name == name, label=f"run_name=='{name}'")

    @classmethod
    def intervention_in(cls, allowed: Iterable[str]) -> RunSpec:
        """Spec admitting manifests whose ``intervention`` is in ``allowed``."""
        allowed_set = frozenset(allowed)
        return cls(
            predicate=lambda m: m.intervention in allowed_set,
            label=f'intervention in {sorted(allowed_set)}',
        )

    @classmethod
    def protocol_eq(cls, protocol: str) -> RunSpec:
        """Spec admitting manifests whose ``protocol`` equals ``protocol``."""
        return cls(predicate=lambda m: m.protocol == protocol, label=f"protocol=='{protocol}'")

    @classmethod
    def min_subset(cls, n: int) -> RunSpec:
        """Spec admitting manifests with ``subset >= n``."""
        return cls(predicate=lambda m: m.subset >= n, label=f'subset>={n}')

    @classmethod
    def dir_must_exist(cls, root: Path) -> RunSpec:
        """Spec admitting manifests whose ``run_dir_path`` exists on disk under ``root``.

        Resolves ``run_dir_path`` relative to ``root`` if not absolute.
        """

        def predicate(manifest: RunManifest) -> bool:
            raw = Path(manifest.run_dir_path)
            resolved = raw if raw.is_absolute() else root / raw
            return resolved.exists() and resolved.is_dir()

        return cls(predicate=predicate, label=f'dir_exists_under({root})')

    @classmethod
    def paper_arm_in(cls, allowed: Iterable[str]) -> RunSpec:
        """Spec admitting manifests whose paper arm is in ``allowed``.

        Resolution honours ``ANALYSIS_CONFIG.paper_arm_by_run_id``; manifests
        with no mapping entry are rejected.
        """
        allowed_set = frozenset(allowed)
        mapping = ANALYSIS_CONFIG.paper_arm_by_run_id
        return cls(
            predicate=lambda m: mapping.get(m.run_id, '') in allowed_set,
            label=f'paper_arm in {sorted(allowed_set)}',
        )


def resolve_paper_arm(manifest: RunManifest) -> str:
    """Return the paper-arm label for ``manifest`` or raise ``KeyError``.

    Looks up ``manifest.run_id`` in ``ANALYSIS_CONFIG.paper_arm_by_run_id``.
    """
    mapping = ANALYSIS_CONFIG.paper_arm_by_run_id
    if manifest.run_id not in mapping:
        raise KeyError(
            f'run_id={manifest.run_id} has no paper-arm mapping in analysis.yaml',
        )
    return mapping[manifest.run_id]


def select_runs(
    manifests: Iterable[RunManifest],
    spec: RunSpec,
) -> tuple[RunManifest, ...]:
    """Filter ``manifests`` through ``spec``, preserving input order."""
    return tuple(m for m in manifests if spec(m))


def _resolve_run_dir(manifest: RunManifest, root: Path) -> Path:
    raw = Path(manifest.run_dir_path)
    return raw if raw.is_absolute() else root / raw


def load_metric_rows(manifest: RunManifest, root: Path) -> pl.DataFrame:
    """Load ``stream_metric_rows.csv`` for one manifest into a Polars dataframe.

    Args:
        manifest: The run manifest.
        root: Workspace root that ``manifest.run_dir_path`` is interpreted
            relative to (when not already absolute).

    Returns:
        Polars dataframe; empty (with no columns) when the CSV is missing.

    Raises:
        FileNotFoundError: If the run directory itself is missing on disk.
    """
    run_dir = _resolve_run_dir(manifest, root)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    csv_path = run_dir / 'stream_metric_rows.csv'
    if not csv_path.exists():
        return pl.DataFrame()
    # Some runs were resumed mid-stream and have the header line repeated
    # inside the body.  Read everything as Utf8 first, drop those repeats,
    # then let polars re-infer dtypes from the cleaned frame.
    raw = pl.read_csv(csv_path, infer_schema_length=0)  # all Utf8
    if raw.is_empty():
        return raw
    cleaned = raw.filter(pl.col('sample_id') != 'sample_id')
    # Round-trip through CSV bytes so polars can infer the proper schema
    # (Int64 / Float64 / etc.) on the cleaned content.
    typed = pl.read_csv(cleaned.write_csv().encode(), infer_schema_length=10000)
    # Pairing key: ``sample_id`` has the form ``sample-<run_local_idx>-<hash16>``
    # where the trailing 16-hex prompt-hash suffix is stable across runs for the
    # same prompt and is therefore the canonical pairing key.
    return typed.with_columns(
        pl.col('sample_id').str.extract(r'-([0-9a-f]{16})$', 1).alias('prompt_hash'),
    ).unique(subset=['prompt_hash'], keep='first')


def load_round_metrics(manifest: RunManifest, root: Path) -> pl.DataFrame:
    """Load ``stream_round_metrics.csv`` (per-turn) for one manifest.

    The per-turn frame carries one row per ``(sample_id, turn_index)`` and
    contains the ``bias_prevalence`` time series used by the lifecycle and
    Markov analyses.

    Args:
        manifest: The run manifest.
        root: Workspace root that ``manifest.run_dir_path`` is interpreted
            relative to.

    Returns:
        Polars dataframe with ``prompt_hash`` derived from ``sample_id``;
        empty (with no columns) when the CSV is missing.

    Raises:
        FileNotFoundError: If the run directory itself is missing on disk.
    """
    run_dir = _resolve_run_dir(manifest, root)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    csv_path = run_dir / 'stream_round_metrics.csv'
    if not csv_path.exists():
        return pl.DataFrame()
    raw = pl.read_csv(csv_path, infer_schema_length=0)
    if raw.is_empty():
        return raw
    cleaned = raw.filter(pl.col('sample_id') != 'sample_id')
    typed = pl.read_csv(cleaned.write_csv().encode(), infer_schema_length=10000)
    return typed.with_columns(
        pl.col('sample_id').str.extract(r'-([0-9a-f]{16})$', 1).alias('prompt_hash'),
    )


def verify_pair_consistency(
    arms: Mapping[str, pl.DataFrame],
    *,
    key: str = 'sample_id',
) -> None:
    """Assert that every arm dataframe contains the same ``key`` set.

    Args:
        arms: Mapping ``arm_name -> dataframe``; each dataframe must
            contain the ``key`` column.
        key: Column whose value-set must match across all arms.

    Raises:
        ValueError: If ``arms`` is empty, any frame lacks ``key``, or
            any pair of arms differs on the ``key`` value-set.
    """
    if not arms:
        raise ValueError('verify_pair_consistency requires at least one arm')
    iterator = iter(arms.items())
    first_name, first_df = next(iterator)
    if key not in first_df.columns:
        raise ValueError(f"arm '{first_name}' is missing key column '{key}'")
    reference = frozenset(first_df.get_column(key).to_list())
    for arm_name, frame in iterator:
        if key not in frame.columns:
            raise ValueError(f"arm '{arm_name}' is missing key column '{key}'")
        candidate = frozenset(frame.get_column(key).to_list())
        if candidate != reference:
            only_first = sorted(reference - candidate)[:3]
            only_other = sorted(candidate - reference)[:3]
            raise ValueError(
                f"pair-consistency violated between '{first_name}' and "
                f"'{arm_name}' on key '{key}': "
                f'first-only sample={only_first}; other-only sample={only_other}',
            )


def intersect_pair_table(
    arms: Mapping[str, pl.DataFrame],
    *,
    key: str = 'sample_id',
) -> dict[str, pl.DataFrame]:
    """Return arm dataframes restricted to the intersection of their ``key`` sets.

    Args:
        arms: Mapping ``arm_name -> dataframe``.
        key: Column to intersect on.

    Returns:
        Mapping ``arm_name -> dataframe`` with each frame filtered (and
        sorted by ``key``) so that all arms share an identical, ordered
        ``key`` column.  An empty input yields an empty mapping.
    """
    if not arms:
        return {}
    key_sets = (frozenset(df.get_column(key).to_list()) for df in arms.values())
    shared = sorted(_reduce_intersection(key_sets))
    return {
        arm_name: df.filter(pl.col(key).is_in(shared)).sort(key) for arm_name, df in arms.items()
    }


def _reduce_intersection(sets: Iterable[frozenset[str]]) -> frozenset[str]:
    iterator = iter(sets)
    try:
        accumulator = next(iterator)
    except StopIteration:
        return frozenset()
    for nxt in iterator:
        accumulator &= nxt
    return accumulator


# Suppress "unused import" warnings when consumers want and_/or_ from operator.
_ = (and_, or_)
