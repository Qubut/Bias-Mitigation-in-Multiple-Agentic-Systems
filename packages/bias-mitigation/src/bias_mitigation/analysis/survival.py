"""Survival-analysis Adapter over ``lifelines`` for the emergence outcome.

The pipeline emits ``MAS_Emergence_Rate`` as a *raw turn index* in
``{0, 1, ..., turn_count - 1}`` (the index of the first turn at which
bias is observed), with the sentinel ``-1.0`` meaning "no bias emerged
within the run". This module re-encodes the column into
a proper *(duration, event_observed)* pair and exposes Adapter
functions for Kaplan-Meier, log-rank, and Cox proportional-hazards
estimators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test as _logrank_test

from bias_mitigation.analysis.config import ANALYSIS_CONFIG

__all__ = [
    'CoxResult',
    'KaplanMeierEstimate',
    'LogRankResult',
    'cox_proportional_hazards',
    'kaplan_meier',
    'logrank',
    're_encode_emergence',
]


CENSORED_SENTINEL: Final[float] = ANALYSIS_CONFIG.censored_sentinel


def re_encode_emergence(
    emergence_rate: Sequence[float] | npt.NDArray[np.float64] | pl.Series,
    turn_count: Sequence[int] | npt.NDArray[np.int64] | pl.Series,
    *,
    censored_sentinel: float = CENSORED_SENTINEL,
) -> pl.DataFrame:
    """Re-encode the emergence-rate column into a survival-analysis pair.

    The pipeline scorer (``mas/metrics.py:emergence_rate``) emits a raw
    turn index in ``{0, 1, ..., turn_count - 1}`` for samples where bias
    was observed, and the sentinel ``censored_sentinel`` (default
    ``-1.0``) when bias never emerged within the run.  This function
    converts that pair into the ``(duration, event_observed)``
    representation expected by ``lifelines``:

    * Event observed (``emergence != sentinel``): ``duration = emergence + 1``
      so that bias appearing at turn 0 corresponds to a duration of one
      time unit ("observed after one turn").
    * Right-censored (``emergence == sentinel``): ``duration = turn_count``,
      i.e. the sample survived the entire observation window.

    Args:
        emergence_rate: Column of raw turn indices; ``censored_sentinel``
            flags a censored observation.
        turn_count: Column with each sample's total number of observed
            turns.  Must have the same length as ``emergence_rate``.
        censored_sentinel: Value used in ``emergence_rate`` to flag a
            censored observation.

    Returns:
        Polars dataframe with columns ``duration`` (float) and
        ``event_observed`` (bool).

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    rates = np.asarray(emergence_rate, dtype=np.float64)
    turns = np.asarray(turn_count, dtype=np.float64)
    if rates.shape != turns.shape:
        raise ValueError(
            f'emergence_rate (shape {rates.shape}) and turn_count (shape {turns.shape}) must match',
        )
    event_observed = rates != censored_sentinel
    duration = np.where(event_observed, rates + 1.0, turns)
    return pl.DataFrame(
        {
            'duration': duration,
            'event_observed': event_observed,
        },
    )


@dataclass(frozen=True, slots=True)
class KaplanMeierEstimate:
    r"""Frozen Kaplan-Meier estimate with timeline and 95 % HDI band.

    Attributes:
        timeline: Monotone-increasing event/censoring times.
        survival: Estimated survival function :math:`\\hat{S}(t)` aligned with ``timeline``.
        ci_lower: Lower bound of the 95 % point-wise confidence band.
        ci_upper: Upper bound of the 95 % point-wise confidence band.
        median_survival_time: Median survival time (``inf`` if not reached).
        n_observed: Total number of observed events.
        n_censored: Total number of right-censored observations.
    """

    timeline: npt.NDArray[np.float64]
    survival: npt.NDArray[np.float64]
    ci_lower: npt.NDArray[np.float64]
    ci_upper: npt.NDArray[np.float64]
    median_survival_time: float
    n_observed: int
    n_censored: int


def kaplan_meier(
    durations: Sequence[float] | npt.NDArray[np.float64] | pl.Series,
    event_observed: Sequence[bool] | npt.NDArray[np.bool_] | pl.Series,
    *,
    label: str = 'KM',
    alpha: float = 0.05,
) -> KaplanMeierEstimate:
    """Fit a Kaplan-Meier estimator and return the immutable estimate.

    Args:
        durations: Time-to-event or time-to-censoring per sample.
        event_observed: ``True`` if the event was observed for that sample.
        label: Label passed to lifelines (purely cosmetic).
        alpha: Significance level for the confidence band; ``0.05`` ⇒ 95 % CI.

    Returns:
        A :class:`KaplanMeierEstimate`.

    References:
        Kaplan & Meier (1958), *Nonparametric estimation from incomplete
        observations*, JASA 53(282).
    """
    durations_arr = np.asarray(durations, dtype=np.float64)
    events_arr = np.asarray(event_observed, dtype=bool)
    fitter = KaplanMeierFitter(alpha=alpha, label=label)
    fitter.fit(durations=durations_arr, event_observed=events_arr)
    sf = fitter.survival_function_
    ci = fitter.confidence_interval_
    return KaplanMeierEstimate(
        timeline=np.asarray(sf.index.to_numpy(), dtype=np.float64),
        survival=np.asarray(sf.iloc[:, 0].to_numpy(), dtype=np.float64),
        ci_lower=np.asarray(ci.iloc[:, 0].to_numpy(), dtype=np.float64),
        ci_upper=np.asarray(ci.iloc[:, 1].to_numpy(), dtype=np.float64),
        median_survival_time=float(fitter.median_survival_time_),
        n_observed=int(events_arr.sum()),
        n_censored=int((~events_arr).sum()),
    )


@dataclass(frozen=True, slots=True)
class LogRankResult:
    """Frozen log-rank two-sample test result.

    Attributes:
        test_statistic: Log-rank chi-square statistic.
        p_value: Two-sided p-value under :math:`H_0` of equal survival curves.
        degrees_of_freedom: Degrees of freedom (``1`` for the two-sample test).
    """

    test_statistic: float
    p_value: float
    degrees_of_freedom: int


def logrank(
    durations_a: Sequence[float] | npt.NDArray[np.float64] | pl.Series,
    events_a: Sequence[bool] | npt.NDArray[np.bool_] | pl.Series,
    durations_b: Sequence[float] | npt.NDArray[np.float64] | pl.Series,
    events_b: Sequence[bool] | npt.NDArray[np.bool_] | pl.Series,
) -> LogRankResult:
    """Two-sample log-rank test for equality of survival functions.

    Args:
        durations_a: Durations for group A.
        events_a: Event indicators for group A.
        durations_b: Durations for group B.
        events_b: Event indicators for group B.

    Returns:
        A :class:`LogRankResult`.
    """
    result = _logrank_test(
        np.asarray(durations_a, dtype=np.float64),
        np.asarray(durations_b, dtype=np.float64),
        event_observed_A=np.asarray(events_a, dtype=bool),
        event_observed_B=np.asarray(events_b, dtype=bool),
    )
    return LogRankResult(
        test_statistic=float(result.test_statistic),
        p_value=float(result.p_value),
        degrees_of_freedom=1,
    )


@dataclass(frozen=True, slots=True)
class CoxCovariate:
    r"""Per-covariate Cox-PH summary row.

    Attributes:
        name: Covariate name.
        coefficient: Estimated regression coefficient :math:`\\beta`.
        hazard_ratio: :math:`\\exp(\\beta)`.
        ci_lower: Lower bound of the 95 % CI for the hazard ratio.
        ci_upper: Upper bound of the 95 % CI for the hazard ratio.
        p_value: Two-sided p-value under :math:`H_0: \\beta = 0`.
    """

    name: str
    coefficient: float
    hazard_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float


@dataclass(frozen=True, slots=True)
class CoxResult:
    """Frozen Cox proportional-hazards model result.

    Attributes:
        covariates: Per-covariate summary tuple.
        concordance_index: Harrell's c-index.
        log_likelihood: Partial log-likelihood at the optimum.
        n_observations: Number of rows entering the fit.
        n_events: Number of observed events.
    """

    covariates: tuple[CoxCovariate, ...]
    concordance_index: float
    log_likelihood: float
    n_observations: int
    n_events: int

    def by_name(self, name: str) -> CoxCovariate:
        """Look up a covariate row by name."""
        for cov in self.covariates:
            if cov.name == name:
                return cov
        raise KeyError(f'no covariate named {name!r}')


def cox_proportional_hazards(
    data: pl.DataFrame,
    *,
    duration_col: str = 'duration',
    event_col: str = 'event_observed',
    covariates: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> CoxResult:
    """Fit a Cox proportional-hazards model and return the immutable result.

    Args:
        data: Polars dataframe; will be converted to pandas internally for
            lifelines compatibility.
        duration_col: Name of the duration column in ``data``.
        event_col: Name of the boolean event-observed column in ``data``.
        covariates: Columns to use as covariates.  ``None`` means use all
            other columns in ``data``.
        alpha: Significance level for the hazard-ratio CI.

    Returns:
        A :class:`CoxResult`.

    References:
        Cox (1972), *Regression models and life tables*, JRSS-B 34(2).
    """
    pdf = data.to_pandas()
    if covariates is not None:
        pdf = pdf[[duration_col, event_col, *covariates]]
    fitter = CoxPHFitter(alpha=alpha)
    fitter.fit(pdf, duration_col=duration_col, event_col=event_col)
    summary = fitter.summary
    ci_lower_col = next(c for c in summary.columns if c.startswith('exp(coef) lower'))
    ci_upper_col = next(c for c in summary.columns if c.startswith('exp(coef) upper'))
    rows = tuple(
        CoxCovariate(
            name=str(name),
            coefficient=float(row['coef']),
            hazard_ratio=float(row['exp(coef)']),
            ci_lower=float(row[ci_lower_col]),
            ci_upper=float(row[ci_upper_col]),
            p_value=float(row['p']),
        )
        for name, row in summary.iterrows()
    )
    return CoxResult(
        covariates=rows,
        concordance_index=float(fitter.concordance_index_),
        log_likelihood=float(fitter.log_likelihood_),
        n_observations=int(pdf.shape[0]),
        n_events=int(pdf[event_col].sum()),
    )
