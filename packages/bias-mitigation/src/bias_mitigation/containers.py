"""Declarative dependency-injector wiring for the MAS runtime.

Design contract:

* :class:`MASConfig` (Pydantic) is the single source of truth.  Cross-field
  invariants (e.g. *memory_config required when intervention is mem0g*)
  are enforced by the model's validators, not by the container — so the
  container only describes wiring, never policy.
* The container declares :class:`MASConfig` as a
  :class:`providers.Dependency` rather than parsing a dict.  Callers pass
  the already-parsed instance via ``Container(mas_config=mas_config)``,
  eliminating the Pydantic→dict→Pydantic round-trip.
* Per-intervention branching uses :class:`providers.Selector` keyed on
  ``mas_config.provided.intervention`` so the runtime mapping is visible
  in the class body instead of buried inside a ``match`` ladder.
* :class:`MASProgram` exposes its own syncify-on-construction named
  constructor (:meth:`MASProgram.syncified`), so the container stays
  ignorant of DSPy adapter details.

Design patterns at a glance (justified in the module docstring tests
discussion in CONTRIBUTING.md):

* **Inversion of Control** — caller owns config construction; container
  owns wiring.  Idiomatic via :class:`providers.Dependency`.
* **Strategy via Selector** — the intervention enum is the strategy key,
  and each branch is a fully-typed provider.  Replaces the ``match`` on
  intervention type.
* **Named Constructor** — :meth:`MASProgram.syncified` keeps the
  DSPy-specific ``cast(MASProgram, syncify(...))`` ritual co-located with
  the class it returns, instead of leaking into the wiring file.
* **Fail Fast** — :func:`_assert_intervention_coverage` runs at import
  time so a future intervention added to :class:`InterventionType` without
  a matching Selector branch raises before any sample is evaluated.

Critique:

* :class:`providers.Selector` does not support a default branch, so all
  four interventions must appear as Selector keys verbatim.  Acceptable —
  adding an intervention is already a multi-file change (config schema,
  protocol map, scripts/CLI choices), and the import-time invariant guard
  makes the missing key a loud failure.
* Two of the four Selector branches are :class:`providers.Object` ``None``
  sentinels.  Repetition is the price of declarativity; the imperative
  alternative re-encoded the same fact as an ``if intervention not in {…}``
  in two places (memory tools + orchestrator) which is no shorter.
"""

from dependency_injector import containers, providers

from bias_mitigation.data.models.config import InterventionType, MASConfig
from bias_mitigation.mas.mas_program import MASProgram
from bias_mitigation.memory.mem0_tools import Mem0Tools
from bias_mitigation.memory.orchestration.service import MemoryOrchestrator


class Container(containers.DeclarativeContainer):
    """MAS runtime DI graph.

    Usage::

        mas_config = MASConfig.model_validate(yaml_payload)
        container = Container(mas_config=mas_config)
        container.wire(packages=['bias_mitigation.mas', 'bias_mitigation.memory'])
        program = container.mas_program()

    For tests, any provider can be overridden::

        container.memory_tools.override(providers.Object(stub))
        try:
            ...
        finally:
            container.memory_tools.reset_override()
    """

    mas_config = providers.Dependency(instance_of=MASConfig)

    memory_tools: providers.Selector = providers.Selector(
        mas_config.provided.intervention,
        baseline=providers.Object(None),
        baseline_opt=providers.Object(None),
        mem0g=providers.Singleton(Mem0Tools, mas_config.provided.memory_config),
        mem0g_gepa=providers.Singleton(Mem0Tools, mas_config.provided.memory_config),
    )

    memory_orchestrator: providers.Selector = providers.Selector(
        mas_config.provided.intervention,
        baseline=providers.Object(None),
        baseline_opt=providers.Object(None),
        mem0g=providers.Singleton(
            MemoryOrchestrator,
            memory_tools=memory_tools,
            config=mas_config.provided.memory_orchestration,
        ),
        mem0g_gepa=providers.Singleton(
            MemoryOrchestrator,
            memory_tools=memory_tools,
            config=mas_config.provided.memory_orchestration,
        ),
    )

    # ``Factory`` (not ``Singleton``) because GEPA needs to deep-copy program
    # candidates per evaluation pass; sharing an instance would conflate
    # mutation across runs.
    mas_program = providers.Factory(
        MASProgram.syncified,
        config=mas_config,
        memory_tools=memory_tools,
        memory_orchestrator=memory_orchestrator,
    )


def _assert_intervention_coverage() -> None:
    """Fail at import time if any :class:`InterventionType` lacks a Selector branch.

    Selector raises ``KeyError`` lazily on the first call for an unknown
    key.  Surfacing that as an import-time assertion keeps drift between
    the enum and the wiring loud and immediate.
    """
    declared = {e.value for e in InterventionType}
    for provider_name in ('memory_tools', 'memory_orchestrator'):
        selector_provider = getattr(Container, provider_name)
        wired = set(selector_provider.providers)
        missing = declared - wired
        if missing:
            raise RuntimeError(
                f'Container.{provider_name} is missing Selector branches for '
                f'InterventionType members: {sorted(missing)}',
            )


_assert_intervention_coverage()
