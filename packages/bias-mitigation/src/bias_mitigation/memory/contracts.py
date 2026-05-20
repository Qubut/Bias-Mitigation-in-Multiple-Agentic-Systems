"""Typed contracts for the mem0-backed memory subsystem.

This module declares the typed shapes that flow in and out of the memory
layer used by agents during the interaction phase. Agents persist their
own prior turns and recall peer history through :class:`Mem0Tools`,
which is bound to :class:`mem0.AsyncMemory` directly — see
:mod:`bias_mitigation.memory.mem0_tools`.

Keeping these contracts narrow lets us swap mem0 for an alternative
vector memory backend (e.g. a Chroma-only implementation) without
touching agent or evaluator code, which is important for reproducibility
experiments where isolating the memory variable matters.
"""

from typing import TypedDict


class MemoryRecord(TypedDict, total=False):
    """A single memory row as returned by a provider's search or get API.

    Mirrors the subset of ``mem0.configs.base.MemoryItem`` that the
    Bias-Mitigation MAS reads — the full mem0 record also carries
    ``hash``, ``metadata``, ``score``, ``created_at``, and ``updated_at``,
    none of which the agent recall path uses.

    Attributes:
        id: Provider-assigned identifier for the row, used for deletion or
            update operations.
        memory: Raw memory text as stored by mem0.
    """

    id: str
    memory: str


class MemoryResultEnvelope(TypedDict):
    """Provider-side envelope wrapping a list of memory rows.

    ``mem0.Memory.search`` and ``mem0.Memory.get_all`` always wrap their
    response payload as ``{"results": [...]}`` (verified against the
    installed mem0 2.0.2 source).  This typed alias is what
    :meth:`Mem0Tools._normalize_search_payload` checks against before
    handing the rows downstream.

    Attributes:
        results: The list of memory rows contained in the envelope.
    """

    results: list[MemoryRecord]


class MemorySearchResult(TypedDict):
    """Normalised search payload returned by ``Mem0Tools.search``.

    This is the *internal* shape every agent sees after provider responses
    have been flattened. Agents inject ``passages`` into the
    ``past_interaction_memory`` field of ``UpdateAnswerWithMemory`` to ground
    their next response on recalled context.

    Attributes:
        passages: Memory texts ranked by the backend, ready to be joined into
            a prompt-friendly block.
        count: Number of passages returned. Equals ``len(passages)`` but is
            kept explicit so callers can short-circuit without materialising
            the list.
    """

    passages: list[str]
    count: int
