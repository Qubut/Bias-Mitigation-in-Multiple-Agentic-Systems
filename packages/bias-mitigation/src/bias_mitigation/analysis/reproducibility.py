"""Reproducibility-pack value objects and per-venue checklist Strategies.

Separates the *fact* of what was recorded (env hash, dataset SHAs,
protocol-lock SHA, RNG seeds) from the *interpretation* against a venue's
checklist (NeurIPS / ACL / ICML).  Each :class:`ChecklistStrategy`
inspects a :class:`ReproducibilityRecord` and returns a structured
:class:`ChecklistResult`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from bias_mitigation.analysis.config import ANALYSIS_CONFIG

__all__ = [
    'MODEL_CARD_SCHEMA_VERSION',
    'ACLChecklist',
    'ChecklistItem',
    'ChecklistResult',
    'ChecklistStrategy',
    'ICMLChecklist',
    'ModelCard',
    'NeurIPSChecklist',
    'ReproducibilityRecord',
    'build_model_card',
    'compose_environment_hash',
    'serialise_record',
]


MODEL_CARD_SCHEMA_VERSION = '1.0.0'
"""Semantic version of the :class:`ModelCard` schema."""


@dataclass(frozen=True, slots=True)
class ReproducibilityRecord:
    """Frozen reproducibility-pack record.

    Attributes:
        protocol_lock_sha: SHA-256 of the analysis-protocol lock.
        environment_hash: SHA-256 of the canonical (Python version,
            platform, package versions) tuple.
        dataset_shas: Mapping ``dataset_name -> sha256`` for every input
            dataset.
        rng_seeds: Mapping ``component_name -> seed`` covering every RNG
            seed used in the analysis pipeline.
        library_versions: Mapping ``package_name -> version`` for the key
            scientific dependencies.
        git_commit: Short git commit SHA at the time of the run, or empty.
        notes: Free-text notes (e.g. known caveats).
    """

    protocol_lock_sha: str
    environment_hash: str
    dataset_shas: Mapping[str, str] = field(default_factory=dict)
    rng_seeds: Mapping[str, int] = field(default_factory=dict)
    library_versions: Mapping[str, str] = field(default_factory=dict)
    git_commit: str = ''
    notes: str = ''


def compose_environment_hash(library_versions: Mapping[str, str]) -> str:
    """Compute a deterministic SHA-256 over the canonical environment tuple.

    Args:
        library_versions: Mapping ``package_name -> version``.

    Returns:
        Hex-encoded SHA-256 covering Python version, platform string, and
        the sorted ``library_versions`` mapping.
    """
    payload = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'libraries': dict(sorted(library_versions.items())),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def serialise_record(record: ReproducibilityRecord) -> str:
    """Serialise a :class:`ReproducibilityRecord` to a canonical JSON string."""
    payload = {
        'protocol_lock_sha': record.protocol_lock_sha,
        'environment_hash': record.environment_hash,
        'dataset_shas': dict(sorted(record.dataset_shas.items())),
        'rng_seeds': dict(sorted(record.rng_seeds.items())),
        'library_versions': dict(sorted(record.library_versions.items())),
        'git_commit': record.git_commit,
        'notes': record.notes,
    }
    return json.dumps(payload, sort_keys=True, indent=2)


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    """Frozen single checklist row.

    Attributes:
        text: Checklist question/requirement text.
        passed: Whether the record satisfies the requirement.
        note: Free-text comment (e.g. evidence pointer or remediation).
    """

    text: str
    passed: bool
    note: str = ''


@dataclass(frozen=True, slots=True)
class ChecklistResult:
    """Frozen collection of :class:`ChecklistItem` plus venue label.

    Attributes:
        venue: Venue tag (``'NeurIPS'``, ``'ACL'``, ``'ICML'``).
        items: Tuple of evaluated items.
    """

    venue: str
    items: tuple[ChecklistItem, ...]

    @property
    def all_passed(self) -> bool:
        """``True`` iff every item passed."""
        return all(item.passed for item in self.items)

    @property
    def fail_count(self) -> int:
        """Number of items that did not pass."""
        return sum(1 for item in self.items if not item.passed)


class ChecklistStrategy(Protocol):
    """Strategy interface for venue-specific checklist evaluation."""

    @property
    def venue(self) -> str: ...

    def evaluate(self, record: ReproducibilityRecord) -> ChecklistResult: ...


def _common_items(record: ReproducibilityRecord) -> list[ChecklistItem]:
    return [
        ChecklistItem(
            text='All RNG seeds are recorded.',
            passed=len(record.rng_seeds) > 0,
            note=f'{len(record.rng_seeds)} seed(s) recorded',
        ),
        ChecklistItem(
            text='Dataset SHA-256s are recorded for every input.',
            passed=len(record.dataset_shas) > 0,
            note=f'{len(record.dataset_shas)} dataset SHA(s) recorded',
        ),
        ChecklistItem(
            text='Environment hash is recorded.',
            passed=bool(record.environment_hash),
            note=f'env_hash={record.environment_hash[:12]}...'
            if record.environment_hash
            else 'missing',
        ),
        ChecklistItem(
            text='Analysis-protocol lock SHA-256 is recorded.',
            passed=bool(record.protocol_lock_sha),
            note=(
                f'protocol_lock={record.protocol_lock_sha[:12]}...'
                if record.protocol_lock_sha
                else 'missing'
            ),
        ),
        ChecklistItem(
            text='Library versions are pinned for the key scientific stack.',
            passed=len(record.library_versions)
            >= ANALYSIS_CONFIG.reproducibility.required_library_count,
            note=f'{len(record.library_versions)} package version(s) pinned',
        ),
    ]


@dataclass(frozen=True, slots=True)
class NeurIPSChecklist:
    """NeurIPS reproducibility-checklist Strategy."""

    @property
    def venue(self) -> str:
        return 'NeurIPS'

    def evaluate(self, record: ReproducibilityRecord) -> ChecklistResult:
        """Evaluate ``record`` against the NeurIPS checklist."""
        items = (
            *_common_items(record),
            ChecklistItem(
                text='Source code is available with a permissive licence.',
                passed=bool(record.git_commit),
                note=f'git={record.git_commit}' if record.git_commit else 'missing',
            ),
        )
        return ChecklistResult(venue=self.venue, items=items)


@dataclass(frozen=True, slots=True)
class ACLChecklist:
    """ACL responsible-NLP-checklist Strategy."""

    @property
    def venue(self) -> str:
        return 'ACL'

    def evaluate(self, record: ReproducibilityRecord) -> ChecklistResult:
        """Evaluate ``record`` against the ACL Responsible NLP checklist."""
        items = (
            *_common_items(record),
            ChecklistItem(
                text='Computational budget is documented in the notes field.',
                passed='compute' in record.notes.lower(),
                note='look for the word "compute" in record.notes',
            ),
        )
        return ChecklistResult(venue=self.venue, items=items)


@dataclass(frozen=True, slots=True)
class ICMLChecklist:
    """ICML reproducibility-checklist Strategy."""

    @property
    def venue(self) -> str:
        return 'ICML'

    def evaluate(self, record: ReproducibilityRecord) -> ChecklistResult:
        """Evaluate ``record`` against the ICML checklist."""
        items = (
            *_common_items(record),
            ChecklistItem(
                text='Git commit is recorded for the analysis pipeline.',
                passed=bool(record.git_commit),
                note=f'git={record.git_commit}' if record.git_commit else 'missing',
            ),
        )
        return ChecklistResult(venue=self.venue, items=items)


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Frozen model card summarising the analysis pipeline.

    Attributes:
        schema_version: Semantic version (``MODEL_CARD_SCHEMA_VERSION``).
        title: Short human-readable title.
        intended_use: One-paragraph statement of intended use.
        out_of_scope: One-paragraph statement of out-of-scope uses.
        primary_outcomes: Tuple of primary outcome metric names.
        arms: Tuple of experimental arm names.
        n_pairs: Sample size of the paired analysis.
        compute_budget: Free-text compute budget (e.g. ``'~6h on a single
            workstation per arm at 3200 samples'``).
        record: The :class:`ReproducibilityRecord` underpinning the card.
        ethical_considerations: Free-text ethical considerations.
    """

    schema_version: str
    title: str
    intended_use: str
    out_of_scope: str
    primary_outcomes: tuple[str, ...]
    arms: tuple[str, ...]
    n_pairs: int
    compute_budget: str
    record: ReproducibilityRecord
    ethical_considerations: str = ''

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a deterministic JSON string."""
        payload = {
            'schema_version': self.schema_version,
            'title': self.title,
            'intended_use': self.intended_use,
            'out_of_scope': self.out_of_scope,
            'primary_outcomes': list(self.primary_outcomes),
            'arms': list(self.arms),
            'n_pairs': self.n_pairs,
            'compute_budget': self.compute_budget,
            'ethical_considerations': self.ethical_considerations,
            'reproducibility': json.loads(serialise_record(self.record)),
        }
        return json.dumps(payload, sort_keys=True, indent=indent)


def build_model_card(  # noqa: C901
    *,
    title: str,
    intended_use: str,
    out_of_scope: str,
    primary_outcomes: tuple[str, ...],
    arms: tuple[str, ...],
    n_pairs: int,
    compute_budget: str,
    record: ReproducibilityRecord,
    ethical_considerations: str = '',
) -> ModelCard:
    """Construct a :class:`ModelCard` with light validation.

    Args:
        title: Non-empty title.
        intended_use: Non-empty intended-use paragraph.
        out_of_scope: Non-empty out-of-scope paragraph.
        primary_outcomes: Non-empty tuple of outcome names.
        arms: Non-empty tuple of arm names.
        n_pairs: Positive sample size.
        compute_budget: Non-empty compute-budget statement.
        record: The :class:`ReproducibilityRecord` underpinning the card.
        ethical_considerations: Optional free-text statement.

    Returns:
        A frozen :class:`ModelCard`.

    Raises:
        ValueError: If any required field is empty or ``n_pairs <= 0``.
    """
    if not title:
        raise ValueError('title must be a non-empty string')
    if not intended_use:
        raise ValueError('intended_use must be a non-empty string')
    if not out_of_scope:
        raise ValueError('out_of_scope must be a non-empty string')
    if not primary_outcomes:
        raise ValueError('primary_outcomes must contain at least one outcome')
    if not arms:
        raise ValueError('arms must contain at least one arm')
    if n_pairs <= 0:
        raise ValueError('n_pairs must be a positive integer')
    if not compute_budget:
        raise ValueError('compute_budget must be a non-empty string')
    return ModelCard(
        schema_version=MODEL_CARD_SCHEMA_VERSION,
        title=title,
        intended_use=intended_use,
        out_of_scope=out_of_scope,
        primary_outcomes=tuple(primary_outcomes),
        arms=tuple(arms),
        n_pairs=n_pairs,
        compute_budget=compute_budget,
        record=record,
        ethical_considerations=ethical_considerations,
    )
