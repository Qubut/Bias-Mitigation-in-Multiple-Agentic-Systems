"""Mixed-effects and GEE Adapter over ``statsmodels`` for paired-arm contrasts.

The 4-arm bias-mitigation study yields a long-format frame with one row per
(``sample_id``, ``arm``) pair.  Naive cross-arm tests violate independence;
the correct estimators are linear mixed models with a random intercept on
``sample_id`` (within-pair correlation) or generalised estimating equations
with an exchangeable working correlation.  This module wraps both as
Adapter functions returning frozen result records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf

if TYPE_CHECKING:
    import pandas as pd  # statsmodels surfaces pandas frames in its result objects.

__all__ = [
    'FixedEffect',
    'GEEResult',
    'MixedLMResult',
    'fit_gee',
    'fit_mixed_lm',
    'paired_arm_contrast',
]


_GEE_FAMILY_FACTORIES: Mapping[str, Callable[[], sm.families.Family]] = {
    'gaussian': sm.families.Gaussian,
    'binomial': sm.families.Binomial,
    'poisson': sm.families.Poisson,
    'gamma': sm.families.Gamma,
}

_GEE_COV_FACTORIES: Mapping[str, Callable[[], sm.cov_struct.CovStruct]] = {
    'exchangeable': sm.cov_struct.Exchangeable,
    'independence': sm.cov_struct.Independence,
    'autoregressive': lambda: sm.cov_struct.Autoregressive(grid=True),
}


@dataclass(frozen=True, slots=True)
class FixedEffect:
    r"""Frozen per-term fixed-effect summary row.

    Attributes:
        name: Term name as emitted by the patsy/formulaic design matrix.
        coefficient: Point estimate :math:`\\hat{\\beta}`.
        std_error: Standard error.
        ci_lower: Lower bound of the 95 % CI.
        ci_upper: Upper bound of the 95 % CI.
        z_value: Wald :math:`z` (or :math:`t`) statistic.
        p_value: Two-sided p-value under :math:`H_0: \\beta = 0`.
    """

    name: str
    coefficient: float
    std_error: float
    ci_lower: float
    ci_upper: float
    z_value: float
    p_value: float


@dataclass(frozen=True, slots=True)
class MixedLMResult:
    """Frozen Linear Mixed Model result.

    Attributes:
        fixed_effects: Per-term fixed-effect rows.
        random_effect_variance: Variance of the random intercept.
        residual_variance: Residual variance estimate.
        log_likelihood: Restricted log-likelihood at convergence.
        aic: Akaike information criterion.
        n_observations: Rows entering the fit.
        n_groups: Number of distinct grouping levels (e.g. pairs).
        converged: Whether the optimiser reported convergence.
    """

    fixed_effects: tuple[FixedEffect, ...]
    random_effect_variance: float
    residual_variance: float
    log_likelihood: float
    aic: float
    n_observations: int
    n_groups: int
    converged: bool

    def by_name(self, name: str) -> FixedEffect:
        """Look up a fixed-effect row by name."""
        for fx in self.fixed_effects:
            if fx.name == name:
                return fx
        raise KeyError(f'no fixed effect named {name!r}')


@dataclass(frozen=True, slots=True)
class GEEResult:
    """Frozen GEE result.

    Attributes:
        fixed_effects: Per-term fixed-effect rows (robust standard errors).
        qic: Quasi-likelihood under the independence model criterion.
        n_observations: Rows entering the fit.
        n_groups: Number of distinct clusters.
        family: Distributional family name.
        cov_structure: Working correlation structure name.
    """

    fixed_effects: tuple[FixedEffect, ...]
    qic: float
    n_observations: int
    n_groups: int
    family: str
    cov_structure: str

    def by_name(self, name: str) -> FixedEffect:
        """Look up a fixed-effect row by name."""
        for fx in self.fixed_effects:
            if fx.name == name:
                return fx
        raise KeyError(f'no fixed effect named {name!r}')


def _summary_to_fixed_effects(
    params: pd.Series,
    bse: pd.Series,
    pvalues: pd.Series,
    conf_int: pd.DataFrame,
) -> tuple[FixedEffect, ...]:
    return tuple(
        FixedEffect(
            name=str(name),
            coefficient=float(params.loc[name]),
            std_error=float(bse.loc[name]),
            ci_lower=float(conf_int.loc[name].iloc[0]),
            ci_upper=float(conf_int.loc[name].iloc[1]),
            z_value=float(params.loc[name] / bse.loc[name]) if bse.loc[name] != 0 else float('nan'),
            p_value=float(pvalues.loc[name]),
        )
        for name in params.index
    )


def fit_mixed_lm(
    data: pl.DataFrame,
    *,
    formula: str,
    groups: str,
    re_formula: str | None = None,
) -> MixedLMResult:
    """Fit a Linear Mixed Model and return the immutable result.

    Args:
        data: Polars dataframe; converted to pandas for statsmodels.
        formula: Patsy-style fixed-effects formula, e.g.
            ``'MAS_System_Robustness ~ C(arm, Treatment(reference="baseline"))'``.
        groups: Column whose levels define the random-effect grouping
            (typically ``'sample_id'`` for paired-arm designs).
        re_formula: Optional patsy formula for additional random effects;
            ``None`` means random intercept only.

    Returns:
        A :class:`MixedLMResult`.

    References:
        Laird & Ware (1982), *Random-effects models for longitudinal data*,
        Biometrics 38(4).
    """
    pdf = data.to_pandas()
    model = smf.mixedlm(formula, pdf, groups=pdf[groups], re_formula=re_formula)
    # Try multiple optimisers; some hit singular Hessians on near-degenerate
    # variance components (e.g. when a metric is near-constant within groups).
    last_exc: Exception | None = None
    fit = None
    for method in ('lbfgs', 'powell', 'cg', 'bfgs'):
        try:
            fit = model.fit(method=method, reml=True)
            break
        except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover - fallback
            last_exc = exc
            continue
    if fit is None:  # pragma: no cover - all optimisers failed
        raise RuntimeError(
            f'MixedLM optimisation failed for all methods: {last_exc}',
        ) from last_exc
    re_var = float(np.asarray(list(fit.cov_re.values))[0][0])
    return MixedLMResult(
        fixed_effects=_summary_to_fixed_effects(
            fit.params.drop('Group Var', errors='ignore'),
            fit.bse.drop('Group Var', errors='ignore'),
            fit.pvalues.drop('Group Var', errors='ignore'),
            fit.conf_int().drop('Group Var', errors='ignore'),
        ),
        random_effect_variance=re_var,
        residual_variance=float(fit.scale),
        log_likelihood=float(fit.llf),
        aic=float(fit.aic),
        n_observations=int(fit.nobs),
        n_groups=int(pdf[groups].nunique()),
        converged=bool(fit.converged),
    )


def fit_gee(
    data: pl.DataFrame,
    *,
    formula: str,
    groups: str,
    family: str = 'gaussian',
    cov_structure: str = 'exchangeable',
) -> GEEResult:
    """Fit a GEE model with cluster-robust SEs and return the immutable result.

    Args:
        data: Polars dataframe; converted to pandas for statsmodels.
        formula: Patsy-style formula.
        groups: Column whose levels define the cluster id.
        family: Distributional family; one of ``'gaussian'``, ``'binomial'``,
            ``'poisson'``, ``'gamma'``.
        cov_structure: Working correlation; one of ``'exchangeable'``,
            ``'independence'``, ``'autoregressive'``.

    Returns:
        A :class:`GEEResult`.

    References:
        Liang & Zeger (1986), *Longitudinal data analysis using generalised
        linear models*, Biometrika 73(1).
    """
    if family not in _GEE_FAMILY_FACTORIES:
        raise ValueError(f'unknown family {family!r}; choose from {sorted(_GEE_FAMILY_FACTORIES)}')
    if cov_structure not in _GEE_COV_FACTORIES:
        raise ValueError(
            f'unknown cov_structure {cov_structure!r}; choose from {sorted(_GEE_COV_FACTORIES)}',
        )
    pdf = data.to_pandas()
    model = smf.gee(
        formula,
        groups=groups,
        data=pdf,
        family=_GEE_FAMILY_FACTORIES[family](),
        cov_struct=_GEE_COV_FACTORIES[cov_structure](),
    )
    fit = model.fit()
    qic_value = fit.qic(scale=fit.scale)
    qic_scalar = float(qic_value[0]) if isinstance(qic_value, tuple) else float(qic_value)
    return GEEResult(
        fixed_effects=_summary_to_fixed_effects(
            fit.params,
            fit.bse,
            fit.pvalues,
            fit.conf_int(),
        ),
        qic=qic_scalar,
        n_observations=int(fit.nobs),
        n_groups=int(pdf[groups].nunique()),
        family=family,
        cov_structure=cov_structure,
    )


def paired_arm_contrast(
    data: pl.DataFrame,
    *,
    outcome: str,
    arm_col: str = 'arm',
    pair_id_col: str = 'sample_id',
    reference_arm: str = 'baseline',
) -> MixedLMResult:
    """Mixed-LM convenience wrapper for the canonical paired-arm contrast.

    Builds the formula
    ``f'{outcome} ~ C({arm_col}, Treatment(reference="{reference_arm}"))'``
    and groups by ``pair_id_col``.

    Args:
        data: Long-format Polars dataframe with one row per (pair, arm).
        outcome: Outcome column name.
        arm_col: Categorical arm column name.
        pair_id_col: Pair identifier column name; becomes the grouping
            variable for the random intercept.
        reference_arm: Level used as the treatment-coding reference.

    Returns:
        A :class:`MixedLMResult` with fixed effects relative to
        ``reference_arm``.
    """
    formula = f'{outcome} ~ C({arm_col}, Treatment(reference="{reference_arm}"))'
    return fit_mixed_lm(data, formula=formula, groups=pair_id_col)
