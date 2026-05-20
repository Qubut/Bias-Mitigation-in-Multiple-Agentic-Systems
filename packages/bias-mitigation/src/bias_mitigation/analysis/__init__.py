"""Post-hoc analysis package — polars pipeline + statistical primitives.

The package surface intentionally stays narrow:

* :class:`LiveAnalysisConfig` + :func:`run_analysis` — the live-runs
  polars pipeline (load → filter → group-bootstrap → write).
* :class:`AnalysisContainer` — DI handle to the lazy heavy
  ``configs/analysis.yaml`` config used by the figure / statistics
  modules.  Tests can ``AnalysisContainer.config.override(...)`` it
  without monkeypatching module-level globals.

Sub-modules (``figures``, ``library_stats``, ``confound``, …) are kept
under the package namespace but **not** re-exported at the top level so
``from bias_mitigation.analysis import …`` does not transitively import
matplotlib/statsmodels/quantecon for callers that only need the runs
pipeline.  Import the submodule you need directly.
"""

from bias_mitigation.analysis.containers import AnalysisContainer
from bias_mitigation.analysis.pipeline import (
    DEFAULT_GROUP_BY,
    METRIC_COLS,
    LiveAnalysisConfig,
    filter_unit_interval,
    group_estimates,
    load_runs,
    run_analysis,
    write_outputs,
)

__all__ = [
    'DEFAULT_GROUP_BY',
    'METRIC_COLS',
    'AnalysisContainer',
    'LiveAnalysisConfig',
    'filter_unit_interval',
    'group_estimates',
    'load_runs',
    'run_analysis',
    'write_outputs',
]
