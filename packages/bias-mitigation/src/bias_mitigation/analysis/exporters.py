"""Master JSON exporter (Builder pattern) for paper-grade analysis artefacts.

Each notebook accumulates intermediate results (tables, model summaries,
sensitivity bands) and finally calls ``MasterJsonBuilder.build()`` to emit
a single, schema-versioned JSON document.  The schema version is bumped
on any breaking change to the document structure so downstream consumers
(LaTeX text, reviewer-checklist tooling) can validate compatibility.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    'MASTER_JSON_SCHEMA_VERSION',
    'MasterJsonArtefact',
    'MasterJsonBuilder',
]

MASTER_JSON_SCHEMA_VERSION = '1.0.0'
"""Semantic version of the master-JSON schema.

Bump the major version on backward-incompatible structural changes
(field removed, type changed); the minor on additive non-breaking
changes; the patch on documentation-only or internal tweaks.
"""


@dataclass(frozen=True, slots=True)
class MasterJsonArtefact:
    """Frozen, schema-versioned master-JSON artefact.

    Attributes:
        schema_version: Semantic schema version (mirrors
            :data:`MASTER_JSON_SCHEMA_VERSION` at build time).
        notebook: Notebook identifier (``'01_paired_main_effects'`` etc.).
        n_pairs: Sample size used by the notebook.
        tables: Mapping ``table_name -> list[dict]`` (each dict is a row).
        scalars: Mapping ``key -> primitive value`` (str | int | float | bool).
        provenance: Reproducibility-pack provenance subset (env hash,
            protocol-lock SHA, library versions, RNG seeds, git commit).
    """

    schema_version: str
    notebook: str
    n_pairs: int
    tables: Mapping[str, list[dict[str, Any]]]
    scalars: Mapping[str, str | int | float | bool]
    provenance: Mapping[str, Any]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a deterministic JSON string.

        Args:
            indent: JSON indent (``None`` for compact).

        Returns:
            UTF-8 JSON string with sorted keys at every level.
        """
        payload = {
            'schema_version': self.schema_version,
            'notebook': self.notebook,
            'n_pairs': self.n_pairs,
            'tables': {k: list(v) for k, v in sorted(self.tables.items())},
            'scalars': dict(sorted(self.scalars.items())),
            'provenance': dict(sorted(self.provenance.items())),
        }
        return json.dumps(payload, sort_keys=True, indent=indent)


@dataclass
class MasterJsonBuilder:
    r"""Builder for :class:`MasterJsonArtefact` (Builder pattern).

    Use the fluent ``add_table`` / ``add_scalar`` / ``set_provenance``
    methods to accumulate state, then call :meth:`build` to validate and
    emit the immutable artefact.

    Attributes:
        notebook: Notebook identifier.
        n_pairs: Sample size used by the notebook.

    Justification of the Builder pattern:
        Notebook code accumulates many heterogeneous fragments
        (tables, scalars, provenance) over dozens of cells; a Builder
        keeps the partial state mutable and validates only at the
        terminal :meth:`build` call.

    Criticism of the Builder pattern:
        Adds boilerplate vs.\ a free dict; partial builds can leak
        invalid state if :meth:`build` is not always called; not
        thread-safe by design.
    """

    notebook: str
    n_pairs: int = 0
    _tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _scalars: dict[str, str | int | float | bool] = field(default_factory=dict)
    _provenance: dict[str, Any] = field(default_factory=dict)

    def add_table(self, name: str, rows: list[dict[str, Any]]) -> MasterJsonBuilder:
        """Add a named table; later calls with the same name overwrite."""
        self._tables[name] = list(rows)
        return self

    def add_scalar(self, name: str, value: str | int | float | bool) -> MasterJsonBuilder:
        """Add a named scalar; later calls with the same name overwrite."""
        self._scalars[name] = value
        return self

    def set_provenance(self, provenance: Mapping[str, Any]) -> MasterJsonBuilder:
        """Set the provenance subset (overwrites any previous setting)."""
        self._provenance = dict(provenance)
        return self

    def set_n_pairs(self, n_pairs: int) -> MasterJsonBuilder:
        """Set the sample size for this artefact."""
        self.n_pairs = n_pairs
        return self

    def build(self) -> MasterJsonArtefact:
        """Validate accumulated state and emit the artefact.

        Returns:
            The frozen :class:`MasterJsonArtefact`.

        Raises:
            ValueError: If ``notebook`` is empty or ``n_pairs <= 0``.
        """
        if not self.notebook:
            raise ValueError('MasterJsonBuilder.notebook must be a non-empty string')
        if self.n_pairs <= 0:
            raise ValueError('MasterJsonBuilder.n_pairs must be > 0 before build()')
        return MasterJsonArtefact(
            schema_version=MASTER_JSON_SCHEMA_VERSION,
            notebook=self.notebook,
            n_pairs=self.n_pairs,
            tables=dict(self._tables),
            scalars=dict(self._scalars),
            provenance=dict(self._provenance),
        )
