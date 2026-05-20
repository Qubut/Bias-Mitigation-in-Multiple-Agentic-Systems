"""Single source of truth for every analysis-stack knob.

Loads ``configs/analysis.yaml`` into a frozen Pydantic tree.  The lazy
singleton :data:`ANALYSIS_CONFIG` is resolved on first access via
:class:`bias_mitigation.analysis.containers.AnalysisContainer` so the
YAML read is deferred until a consumer actually needs a value — earlier
versions invoked :func:`load_analysis_config` at module import and
broke imports in checkouts where the file was absent.

Override path with the ``BIAS_MITIGATION_ANALYSIS_CONFIG`` environment
variable when running tests against a custom config.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    ANALYSIS_CONFIG: AnalysisConfig

__all__ = [
    'ANALYSIS_CONFIG',
    'AnalysisConfig',
    'BootstrapDefaults',
    'CoxDefaults',
    'DefaultsConfig',
    'FiguresConfig',
    'IpwDefaults',
    'KmDefaults',
    'PaletteConfig',
    'ReproducibilityConfig',
    'load_analysis_config',
]


_ENV_VAR: Final[str] = 'BIAS_MITIGATION_ANALYSIS_CONFIG'


def _frozen_model_config() -> ConfigDict:
    return ConfigDict(frozen=True, extra='forbid')


class BootstrapDefaults(BaseModel):
    """Default hyper-parameters for any bootstrap procedure."""

    model_config = _frozen_model_config()
    n_bootstrap: int = Field(ge=1)
    seed: int
    alpha: float = Field(gt=0.0, lt=1.0)


class IpwDefaults(BaseModel):
    """Default hyper-parameters for IPW estimators."""

    model_config = _frozen_model_config()
    clip_low: float = Field(gt=0.0, lt=1.0)
    clip_high: float = Field(gt=0.0, lt=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)


class CoxDefaults(BaseModel):
    """Default hyper-parameters for Cox proportional-hazards fits."""

    model_config = _frozen_model_config()
    alpha: float = Field(gt=0.0, lt=1.0)


class KmDefaults(BaseModel):
    """Default hyper-parameters for Kaplan-Meier fits."""

    model_config = _frozen_model_config()
    alpha: float = Field(gt=0.0, lt=1.0)


class DefaultsConfig(BaseModel):
    """Aggregate of estimator-default sub-blocks."""

    model_config = _frozen_model_config()
    bootstrap: BootstrapDefaults
    ipw: IpwDefaults
    cox: CoxDefaults
    km: KmDefaults


class PaletteConfig(BaseModel):
    """Frozen colour-palette specification."""

    model_config = _frozen_model_config()
    sequential: str
    diverging: str
    arm_colours: Mapping[str, str]


class FiguresConfig(BaseModel):
    """Matplotlib presentation parameters."""

    model_config = _frozen_model_config()
    sigir_double_column_inches: float
    sigir_single_column_inches: float
    figure_dpi: int
    savefig_dpi: int
    font_size: float
    axes_title_size: float
    axes_label_size: float
    tick_label_size: float
    legend_font_size: float
    palette: PaletteConfig


class ReproducibilityConfig(BaseModel):
    """Reproducibility-pack environment knobs."""

    model_config = _frozen_model_config()
    ld_library_path: str
    required_library_count: int = Field(ge=0)


class AnalysisConfig(BaseModel):
    """Top-level frozen analysis-stack configuration tree."""

    model_config = _frozen_model_config()
    metric_columns: tuple[str, ...]
    metric_supports: Mapping[str, tuple[float, float]]
    intervention_display_labels: Mapping[str, str]
    allowed_interventions: tuple[str, ...]
    allowed_protocols: tuple[str, ...]
    cooperative_run_name: str
    paper_arm_by_run_id: Mapping[str, str]
    paper_arms_main: tuple[str, ...]
    paper_arm_full_dev: str
    censored_sentinel: float
    defaults: DefaultsConfig
    figures: FiguresConfig
    reproducibility: ReproducibilityConfig


def _default_config_path() -> Path:
    env_override = os.environ.get(_ENV_VAR)
    if env_override:
        return Path(env_override)
    package_root = Path(__file__).resolve().parent.parent.parent.parent
    return package_root / 'configs' / 'analysis.yaml'


def load_analysis_config(path: Path | None = None) -> AnalysisConfig:
    """Load and validate ``configs/analysis.yaml`` (or override ``path``).

    Args:
        path: Optional path to a YAML file.  When ``None``, honours the
            ``BIAS_MITIGATION_ANALYSIS_CONFIG`` env var, then falls back to
            the bundled ``configs/analysis.yaml`` next to the package root.

    Returns:
        Validated :class:`AnalysisConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML payload violates the schema.
    """
    actual = path or _default_config_path()
    if not actual.exists():
        raise FileNotFoundError(actual)
    with actual.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    return AnalysisConfig.model_validate(data)


def __getattr__(name: str) -> Any:
    """PEP-562 lazy resolver for :data:`ANALYSIS_CONFIG`.

    Defers ``configs/analysis.yaml`` loading until the first attribute
    access on :data:`ANALYSIS_CONFIG`, routed through the analysis DI
    container so tests can ``container.config.override(...)`` it.
    """
    if name == 'ANALYSIS_CONFIG':
        from bias_mitigation.analysis.containers import AnalysisContainer  # noqa: PLC0415
        return AnalysisContainer.config()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
