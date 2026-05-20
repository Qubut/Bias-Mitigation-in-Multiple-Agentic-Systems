"""Confounder-sensitivity Adapter: Manski bounds, IPW, negative-control.

The paired-arm design fully identifies the average treatment effect when
randomisation between arms is intact.  This module provides three
sensitivity tools used for robustness checks in the supplementary
notebook:

* **Manski (1990) bounds**: partial-identification interval on the
  ATE that uses *no* assumption beyond knowing the outcome's support.
  Useful for cross-sectional comparisons where the paired structure has
  been broken (e.g. per-category subsets with unequal arm sizes).

* **Inverse-Probability Weighting (IPW)**: re-weights observations by
  the inverse propensity to treat, with optional weight clipping.

* **Negative-control outcome test**: if the intervention is exchangeable
  with respect to a known invariant outcome, a non-zero estimated effect
  on that outcome flags residual confounding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import stats

from bias_mitigation.analysis.config import ANALYSIS_CONFIG

__all__ = [
    'METRIC_SUPPORTS',
    'IpwAteEstimate',
    'ManskiBounds',
    'NegativeControlResult',
    'inverse_probability_weights',
    'ipw_ate',
    'manski_bounds',
    'negative_control_test',
]


METRIC_SUPPORTS: Final[Mapping[str, tuple[float, float]]] = ANALYSIS_CONFIG.metric_supports


@dataclass(frozen=True, slots=True)
class ManskiBounds:
    r"""Frozen Manski (1990) partial-identification bounds on the ATE.

    Attributes:
        lower: Lower bound on :math:`E[Y(1) - Y(0)]`.
        upper: Upper bound on :math:`E[Y(1) - Y(0)]`.
        width: ``upper - lower``; equals the support width when both
            arms share the same propensity, irrespective of effect size.
        support_min: Assumed lower bound of the outcome's support.
        support_max: Assumed upper bound of the outcome's support.
        n_treated: Number of observations with treatment indicator ``True``.
        n_control: Number of observations with treatment indicator ``False``.
    """

    lower: float
    upper: float
    width: float
    support_min: float
    support_max: float
    n_treated: int
    n_control: int


def manski_bounds(
    outcome: Sequence[float] | npt.NDArray[np.float64],
    treatment: Sequence[bool] | npt.NDArray[np.bool_],
    *,
    support: tuple[float, float],
) -> ManskiBounds:
    """Compute Manski no-assumption bounds on the average treatment effect.

    Args:
        outcome: Continuous outcome values, observed once per sample.
        treatment: Boolean treatment indicators aligned with ``outcome``.
        support: ``(y_min, y_max)`` known support of the outcome variable.
            Use :data:`METRIC_SUPPORTS` for the canonical bias metrics.

    Returns:
        A :class:`ManskiBounds`.

    Raises:
        ValueError: If shapes mismatch, support is degenerate, or either
            treatment arm is empty.

    References:
        Manski (1990), *Nonparametric bounds on treatment effects*,
        American Economic Review 80(2).
    """
    y = np.asarray(outcome, dtype=np.float64)
    t = np.asarray(treatment, dtype=bool)
    if y.shape != t.shape:
        raise ValueError(f'outcome shape {y.shape} != treatment shape {t.shape}')
    y_min, y_max = support
    if not y_max > y_min:
        raise ValueError(f'support upper {y_max} must exceed lower {y_min}')
    n = y.size
    n_treated = int(t.sum())
    n_control = int(n - n_treated)
    if n_treated == 0 or n_control == 0:
        raise ValueError('both treatment arms must be non-empty')
    p_treated = n_treated / n
    p_control = n_control / n
    mean_y_treated = float(y[t].mean())
    mean_y_control = float(y[~t].mean())
    lower = (mean_y_treated * p_treated + y_min * p_control) - (
        mean_y_control * p_control + y_max * p_treated
    )
    upper = (mean_y_treated * p_treated + y_max * p_control) - (
        mean_y_control * p_control + y_min * p_treated
    )
    return ManskiBounds(
        lower=lower,
        upper=upper,
        width=upper - lower,
        support_min=y_min,
        support_max=y_max,
        n_treated=n_treated,
        n_control=n_control,
    )


def inverse_probability_weights(
    treatment: Sequence[bool] | npt.NDArray[np.bool_],
    propensity: Sequence[float] | npt.NDArray[np.float64],
    *,
    clip: tuple[float, float] = (0.05, 0.95),
) -> npt.NDArray[np.float64]:
    r"""Compute clipped inverse-probability-of-treatment weights.

    Args:
        treatment: Boolean treatment indicators.
        propensity: Estimated :math:`P(T=1 \\mid X)` per observation,
            same length as ``treatment``.
        clip: ``(low, high)`` bounds; propensity values are first clipped
            to this interval to control variance from extreme weights.

    Returns:
        Numpy array of weights ``1/p`` for treated rows and
        ``1/(1-p)`` for control rows.

    Raises:
        ValueError: If shapes mismatch, ``clip`` is degenerate, or any
            clipped propensity falls outside ``(0, 1)``.
    """
    t = np.asarray(treatment, dtype=bool)
    p = np.asarray(propensity, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f'treatment shape {t.shape} != propensity shape {p.shape}')
    low, high = clip
    if not 0.0 < low < high < 1.0:
        raise ValueError(f'clip bounds {clip} must satisfy 0 < low < high < 1')
    p_clipped = np.clip(p, low, high)
    return np.where(t, 1.0 / p_clipped, 1.0 / (1.0 - p_clipped))


@dataclass(frozen=True, slots=True)
class IpwAteEstimate:
    r"""Frozen IPW Average-Treatment-Effect estimate.

    Attributes:
        ate: Weighted-mean difference :math:`\\hat{E}[Y(1) - Y(0)]`.
        std_error: Standard error of the weighted mean difference.
        ci_lower: Lower bound of the 95 % CI under a normal approximation.
        ci_upper: Upper bound of the 95 % CI under a normal approximation.
        effective_sample_size: Kish's effective sample size summed across arms.
    """

    ate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    effective_sample_size: float


def ipw_ate(
    outcome: Sequence[float] | npt.NDArray[np.float64],
    treatment: Sequence[bool] | npt.NDArray[np.bool_],
    propensity: Sequence[float] | npt.NDArray[np.float64],
    *,
    clip: tuple[float, float] = (0.05, 0.95),
    alpha: float = 0.05,
) -> IpwAteEstimate:
    r"""Estimate the ATE via Horvitz-Thompson inverse-probability weighting.

    Args:
        outcome: Outcome values per sample.
        treatment: Boolean treatment indicators.
        propensity: Estimated :math:`P(T=1 \\mid X)` per observation.
        clip: Propensity-score clipping bounds passed to
            :func:`inverse_probability_weights`.
        alpha: Significance level for the normal-approximation CI.

    Returns:
        An :class:`IpwAteEstimate`.
    """
    y = np.asarray(outcome, dtype=np.float64)
    t = np.asarray(treatment, dtype=bool)
    weights = inverse_probability_weights(t, propensity, clip=clip)
    weighted_treated = (weights * y * t).sum() / (weights * t).sum()
    weighted_control = (weights * y * ~t).sum() / (weights * ~t).sum()
    ate = float(weighted_treated - weighted_control)
    # Sandwich-style variance approximation
    contributions = np.where(
        t,
        weights * (y - weighted_treated),
        -weights * (y - weighted_control),
    )
    se = float(np.sqrt(np.var(contributions, ddof=1) / y.size))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ess = float((weights.sum() ** 2) / (weights**2).sum())
    return IpwAteEstimate(
        ate=ate,
        std_error=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        effective_sample_size=ess,
    )


@dataclass(frozen=True, slots=True)
class NegativeControlResult:
    """Frozen negative-control-outcome test result.

    Attributes:
        estimated_effect: Mean-difference estimate on the negative-control
            outcome.
        std_error: Standard error of the estimate.
        p_value: Two-sided p-value under :math:`H_0:` no effect.
        passes: ``True`` iff ``p_value > alpha`` (i.e. no detectable effect,
            consistent with the no-confounding hypothesis).
        alpha: Significance threshold.
    """

    estimated_effect: float
    std_error: float
    p_value: float
    passes: bool
    alpha: float


def negative_control_test(
    neg_outcome: Sequence[float] | npt.NDArray[np.float64],
    treatment: Sequence[bool] | npt.NDArray[np.bool_],
    *,
    alpha: float = 0.05,
) -> NegativeControlResult:
    """Welch two-sample test on a negative-control outcome.

    Args:
        neg_outcome: Outcome that is, by design, invariant to the
            intervention (e.g. a fairness metric on a held-out
            non-bias-relevant category).
        treatment: Boolean treatment indicators.
        alpha: Significance threshold for ``passes``.

    Returns:
        A :class:`NegativeControlResult`.  ``passes=True`` indicates that
        no detectable effect was found, which is the *desired* outcome
        for the sensitivity check.
    """
    y = np.asarray(neg_outcome, dtype=np.float64)
    t = np.asarray(treatment, dtype=bool)
    treated = y[t]
    control = y[~t]
    test = stats.ttest_ind(treated, control, equal_var=False)
    estimate = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size))
    p_value = float(test.pvalue)
    return NegativeControlResult(
        estimated_effect=estimate,
        std_error=se,
        p_value=p_value,
        passes=p_value > alpha,
        alpha=alpha,
    )
