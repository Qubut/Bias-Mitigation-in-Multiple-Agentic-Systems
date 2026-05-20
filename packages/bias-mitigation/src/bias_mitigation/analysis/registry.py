"""Hypothesis registry and analysis-protocol SHA-256 lock.

The registry pins primary/secondary outcome metadata (direction, alpha,
multiple-comparison family) into a frozen, hashable structure so that
notebook code cannot silently re-interpret a hypothesis between runs.
The protocol-lock SHA-256 is the cryptographic counterpart: any change
to the registry contents (or to additional designated cell payloads)
shifts the lock, and the pre-commit hook refuses the commit unless the
revision log is updated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    'Direction',
    'Hypothesis',
    'HypothesisRegistry',
    'protocol_lock',
]


Direction = Literal['higher_is_better', 'lower_is_better', 'composition']


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Frozen hypothesis record.

    Attributes:
        name: Human-readable identifier (e.g. ``'H1_Robustness'``).
        outcome: Outcome variable column name (e.g. ``'MAS_System_Robustness'``).
        direction: Whether higher / lower values indicate the favourable
            condition, or whether the outcome is a multinomial composition
            (e.g. trajectory taxonomy) where direction is undefined.
        family: Multiple-comparison family label; tests sharing a family
            are corrected jointly via Benjamini-Hochberg.
        alpha: Per-test significance threshold prior to correction.
        is_primary: ``True`` for confirmatory primary outcomes, ``False``
            for secondary / sensitivity outcomes.
        notes: Free-text rationale captured into the protocol lock.
    """

    name: str
    outcome: str
    direction: Direction
    family: str
    alpha: float = 0.05
    is_primary: bool = True
    notes: str = ''


@dataclass(frozen=True, slots=True)
class HypothesisRegistry:
    """Immutable mapping from hypothesis name to :class:`Hypothesis` record."""

    hypotheses: tuple[Hypothesis, ...]

    def __post_init__(self) -> None:
        names = [h.name for h in self.hypotheses]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f'duplicate hypothesis names: {sorted(duplicates)}')

    def by_name(self, name: str) -> Hypothesis:
        """Return the hypothesis whose ``name`` field equals ``name``."""
        for h in self.hypotheses:
            if h.name == name:
                return h
        raise KeyError(f'no hypothesis named {name!r}')

    def primary(self) -> tuple[Hypothesis, ...]:
        """Return the tuple of primary hypotheses, preserving registration order."""
        return tuple(h for h in self.hypotheses if h.is_primary)

    def families(self) -> Mapping[str, tuple[Hypothesis, ...]]:
        """Group hypotheses by their multiple-comparison family."""
        grouped: dict[str, list[Hypothesis]] = {}
        for h in self.hypotheses:
            grouped.setdefault(h.family, []).append(h)
        return {family: tuple(items) for family, items in grouped.items()}

    def to_payload(self) -> tuple[dict[str, str | float | bool], ...]:
        """Serialise to a deterministic, JSON-friendly payload."""
        return tuple(
            {
                'name': h.name,
                'outcome': h.outcome,
                'direction': h.direction,
                'family': h.family,
                'alpha': h.alpha,
                'is_primary': h.is_primary,
                'notes': h.notes,
            }
            for h in self.hypotheses
        )


_HASH_ENCODING: Final[str] = 'utf-8'


def protocol_lock(
    registry: HypothesisRegistry,
    *,
    extra_payloads: Iterable[Mapping[str, object]] = (),
) -> str:
    """Compute the SHA-256 lock over the registry plus any extra payloads.

    The lock is deterministic across runs: payloads are serialised with
    sorted keys and no insignificant whitespace.  Any change to the
    registry contents or to ``extra_payloads`` shifts the lock value.

    Args:
        registry: The :class:`HypothesisRegistry` to hash.
        extra_payloads: Additional JSON-serialisable mappings to fold into
            the lock (e.g. dataset SHAs, code-cell payloads tagged
            ``prereg`` in the primary notebook).

    Returns:
        Hex-encoded SHA-256 digest as a string.
    """
    hasher = hashlib.sha256()
    canonical_registry = json.dumps(
        list(registry.to_payload()),
        sort_keys=True,
        separators=(',', ':'),
    )
    hasher.update(canonical_registry.encode(_HASH_ENCODING))
    for payload in extra_payloads:
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
        hasher.update(canonical.encode(_HASH_ENCODING))
    return hasher.hexdigest()
