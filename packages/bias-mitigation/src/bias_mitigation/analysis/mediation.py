"""Paired-bootstrap mediation Adapter (Imai-Keele-Tingley 2010).

Causal model: ``Treatment -> Mediator -> Outcome`` (with possible direct
``Treatment -> Outcome`` arc).  In the 4-arm bias-mitigation design the
treatment is binary (intervention vs. baseline) and the data are paired
on ``sample_id``: every prompt is observed under both arms.  This module
implements the difference-on-pairs estimator and a paired non-parametric
bootstrap to obtain bias-corrected confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from bias_mitigation.analysis.config import ANALYSIS_CONFIG

__all__ = [
    'MediationResult',
    'paired_bootstrap_mediation',
]


_BOOTSTRAP_DEFAULTS = ANALYSIS_CONFIG.defaults.bootstrap


@dataclass(frozen=True, slots=True)
class MediationResult:
    """Frozen mediation-analysis result.

    Attributes:
        total_effect: Estimated total effect on the outcome (mean of paired
            differences in ``outcome``).
        indirect_effect: Estimated Average Causal Mediation Effect (ACME):
            slope of ``Δoutcome ~ Δmediator`` times mean ``Δmediator``.
        direct_effect: Estimated Average Direct Effect (ADE): total minus
            indirect.
        proportion_mediated: ``indirect_effect / total_effect`` (``nan``
            when the total effect is zero).
        ci_total: 95 % bootstrap CI on ``total_effect``.
        ci_indirect: 95 % bootstrap CI on ``indirect_effect``.
        ci_direct: 95 % bootstrap CI on ``direct_effect``.
        n_pairs: Number of pairs entering the fit.
        n_bootstrap: Number of bootstrap replicates.
        seed: RNG seed actually used (echoed for reproducibility).
    """

    total_effect: float
    indirect_effect: float
    direct_effect: float
    proportion_mediated: float
    ci_total: tuple[float, float]
    ci_indirect: tuple[float, float]
    ci_direct: tuple[float, float]
    n_pairs: int
    n_bootstrap: int
    seed: int


def _ols_slope(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Closed-form OLS slope of ``y ~ x`` (intercept fit)."""
    x_centred = x - x.mean()
    y_centred = y - y.mean()
    denom = float((x_centred * x_centred).sum())
    if np.abs(denom) < 1e-12:
        return 0.0
    return float((x_centred * y_centred).sum() / denom)


def _mediation_estimates(
    delta_m: npt.NDArray[np.float64],
    delta_y: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    total = float(delta_y.mean())
    slope = _ols_slope(delta_m, delta_y)
    indirect = slope * float(delta_m.mean())
    direct = total - indirect
    return total, indirect, direct


def paired_bootstrap_mediation(
    data: pl.DataFrame,
    *,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    pair_id_col: str = 'sample_id',
    treated_value: object = True,
    control_value: object = False,
    n_bootstrap: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
) -> MediationResult:
    """Estimate mediation effects on paired data via non-parametric bootstrap.

    The dataframe must be in long format with two rows per pair, one per
    arm.  Pairs missing either arm are silently dropped.

    Args:
        data: Long-format Polars dataframe.
        treatment_col: Column distinguishing the two arms.
        mediator_col: Continuous mediator column.
        outcome_col: Continuous outcome column.
        pair_id_col: Pair identifier (typically ``'sample_id'``).
        treated_value: Value of ``treatment_col`` denoting the treated arm.
        control_value: Value of ``treatment_col`` denoting the control arm.
        n_bootstrap: Number of bootstrap replicates over pairs.
        seed: RNG seed for reproducibility.
        alpha: Significance level for the percentile CI; ``0.05`` ⇒ 95 %.

    Returns:
        A :class:`MediationResult`.

    References:
        Imai, Keele & Tingley (2010), *A general approach to causal mediation
        analysis*, Psychological Methods 15(4).
    """
    n_bootstrap_eff = _BOOTSTRAP_DEFAULTS.n_bootstrap if n_bootstrap is None else n_bootstrap
    seed_eff = _BOOTSTRAP_DEFAULTS.seed if seed is None else seed
    alpha_eff = _BOOTSTRAP_DEFAULTS.alpha if alpha is None else alpha
    treated = (
        data
        .filter(pl.col(treatment_col) == treated_value)
        .select(pair_id_col, mediator_col, outcome_col)
        .rename({mediator_col: 'm_t', outcome_col: 'y_t'})
    )
    control = (
        data
        .filter(pl.col(treatment_col) == control_value)
        .select(pair_id_col, mediator_col, outcome_col)
        .rename({mediator_col: 'm_c', outcome_col: 'y_c'})
    )
    paired = treated.join(control, on=pair_id_col, how='inner')
    delta_m = (paired.get_column('m_t') - paired.get_column('m_c')).to_numpy().astype(np.float64)
    delta_y = (paired.get_column('y_t') - paired.get_column('y_c')).to_numpy().astype(np.float64)
    n_pairs = delta_m.size
    if n_pairs < 2:
        raise ValueError(f'need >= 2 complete pairs; got {n_pairs}')
    total, indirect, direct = _mediation_estimates(delta_m, delta_y)
    rng = np.random.default_rng(seed=seed_eff)
    boot_indices = rng.integers(low=0, high=n_pairs, size=(n_bootstrap_eff, n_pairs))
    boot_totals = np.empty(n_bootstrap_eff, dtype=np.float64)
    boot_indirect = np.empty(n_bootstrap_eff, dtype=np.float64)
    boot_direct = np.empty(n_bootstrap_eff, dtype=np.float64)
    for b in range(n_bootstrap_eff):
        idx = boot_indices[b]
        bt, bi, bd = _mediation_estimates(delta_m[idx], delta_y[idx])
        boot_totals[b] = bt
        boot_indirect[b] = bi
        boot_direct[b] = bd
    lo, hi = 100.0 * (alpha_eff / 2.0), 100.0 * (1.0 - alpha_eff / 2.0)
    proportion = indirect / total if np.abs(total) > 1e-12 else float('nan')
    return MediationResult(
        total_effect=total,
        indirect_effect=indirect,
        direct_effect=direct,
        proportion_mediated=proportion,
        ci_total=(float(np.percentile(boot_totals, lo)), float(np.percentile(boot_totals, hi))),
        ci_indirect=(
            float(np.percentile(boot_indirect, lo)),
            float(np.percentile(boot_indirect, hi)),
        ),
        ci_direct=(float(np.percentile(boot_direct, lo)), float(np.percentile(boot_direct, hi))),
        n_pairs=n_pairs,
        n_bootstrap=n_bootstrap_eff,
        seed=seed_eff,
    )
