"""Dependency-injector wiring for the analysis package.

A single :class:`providers.Singleton` defers ``configs/analysis.yaml``
loading until first access — replacing the previous module-level
``ANALYSIS_CONFIG = load_analysis_config()`` line that raised
``FileNotFoundError`` at import time in checkouts without the YAML.

The container also gives tests the standard
:meth:`providers.Singleton.override` seam, matching the MAS-side
:class:`bias_mitigation.containers.Container` pattern.
"""

from dependency_injector import containers, providers

from bias_mitigation.analysis.config import load_analysis_config


class AnalysisContainer(containers.DeclarativeContainer):
    """Lazy DI graph for the analysis stack.

    Usage::

        from bias_mitigation.analysis.containers import AnalysisContainer
        cfg = AnalysisContainer.config()         # loads YAML on first call

    Tests::

        from dependency_injector import providers
        AnalysisContainer.config.override(providers.Object(stub_config))
        try:
            ...
        finally:
            AnalysisContainer.config.reset_override()
    """

    config = providers.Singleton(load_analysis_config)
