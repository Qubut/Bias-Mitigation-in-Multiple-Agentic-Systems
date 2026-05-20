"""Library-backed statistical primitives for MAS bias-mitigation analysis.

The module exposes a small, uniform API over four scientific libraries
(``arviz``, ``scipy.stats``, ``pingouin``, ``quantecon``) so that downstream
analysis code remains backend-agnostic and citation-traceable.  Every
primitive returns either a numeric scalar, a tuple of scalars, or a frozen
result record (:class:`CIResult` / :class:`MarkovSummary`); none mutate
their inputs.

The active backend versions (resolved at import time) are exported as
:data:`BACKEND_VERSIONS` so that any reported result can be paired with
the exact library provenance used to compute it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

import arviz as az
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st
from numpy.typing import NDArray

try:
    import quantecon as qe

    _HAS_QE: Final[bool] = True
    _QE_BACKEND_NAME: Final[str] = 'quantecon.MarkovChain'
except ImportError:
    qe = None
    _HAS_QE = False
    _QE_BACKEND_NAME = 'numpy.linalg.eig (fallback)'


__all__ = [
    'BACKEND_VERSIONS',
    'BHFDRResult',
    'CIResult',
    'MarkovSummary',
    'PairedDiffCI',
    'anova_two_way',
    'bh_fdr',
    'cliffs_delta',
    'cohens_d_paired',
    'hdi_of',
    'hedges_g_paired',
    'markov_chain_summary',
    'paired_diff_bootstrap_ci',
    'permutation_pvalue',
    'safe_bootstrap_mean_ci',
    'safe_wilcoxon_p',
    'transition_counts',
]


BACKEND_VERSIONS: Final[dict[str, str]] = {
    'arviz': getattr(az, '__version__', 'unknown'),
    'pingouin': getattr(pg, '__version__', 'unknown'),
    'scipy': getattr(st, '__name__', 'scipy.stats'),
    'quantecon': qe.__version__ if _HAS_QE and hasattr(qe, '__version__') else 'not-installed',
}


def hdi_of(
    post: Sequence[float],
    hdi_prob: float = 0.95,
    multimodal: bool = False,
) -> tuple[float, float]:
    """Compute the Highest-Density Interval of a posterior sample.

    Args:
        post: 1-D posterior draws. ``NaN`` entries are dropped.
        hdi_prob: Target probability mass of the interval, in ``(0, 1)``.
        multimodal: If ``True``, request multimodal interval detection
            from the backend; the widest reported interval is returned.

    Returns:
        ``(lo, hi)`` such that the interval contains ``hdi_prob`` of the
        posterior mass. Returns ``(nan, nan)`` if the cleaned sample is
        empty.

    References:
        Kruschke, *Doing Bayesian Data Analysis*, 2nd ed., Ch. 25.
    """
    cleaned = np.asarray(post, dtype=float)
    cleaned = cleaned[~np.isnan(cleaned)]
    if cleaned.size == 0:
        return float('nan'), float('nan')
    method: Final[str] = 'multimodal_sample' if multimodal else 'nearest'
    try:
        out = az.hdi(cleaned, prob=hdi_prob, method=method)
    except TypeError:  # pragma: no cover — pre-arviz_stats fallback
        out = az.hdi(cleaned, hdi_prob=hdi_prob, multimodal=multimodal)
    arr = np.asarray(out)
    if arr.ndim == 1:
        return float(arr[0]), float(arr[1])
    widest = int(np.argmax(arr[:, 1] - arr[:, 0]))
    return float(arr[widest, 0]), float(arr[widest, 1])


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    r"""Compute Cliff's :math:`\delta` non-parametric effect size.

    Uses the Mann-Whitney identity
    :math:`\delta = 2\,U_1 / (n_a n_b) - 1`, which avoids the
    :math:`O(n^2)` pairwise-comparison form.

    Args:
        a: First sample. ``NaN`` entries are dropped.
        b: Second sample. ``NaN`` entries are dropped.

    Returns:
        :math:`\delta \in [-1, +1]`. Positive values mean ``a`` stochastically
        dominates ``b``. Returns ``nan`` if either cleaned sample is empty.

    References:
        Cliff, N. (1993). Dominance statistics: ordinal analyses to answer
        ordinal questions. *Psychological Bulletin*, 114(3), 494-509.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = np.asarray(b, dtype=float)
    arr_b = arr_b[~np.isnan(arr_b)]
    if arr_a.size == 0 or arr_b.size == 0:
        return float('nan')
    u, _ = st.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
    return float(2.0 * u / (arr_a.size * arr_b.size) - 1.0)


@dataclass(frozen=True, slots=True)
class MarkovSummary:
    """Frozen summary of a finite-state discrete-time Markov chain.

    Attributes:
        P: Row-stochastic transition matrix.
        pi: Stationary distribution; sums to 1.
        n_transitions: Number of transitions used to estimate ``P``.
        is_ergodic: ``True`` if the chain is irreducible (and therefore
            has a unique stationary distribution); ``None`` if the active
            backend cannot decide.
        backend: Identifier of the library used to compute ``pi``.
        states: Human-readable labels for the rows / columns of ``P``.
    """

    P: NDArray[np.float64]
    pi: NDArray[np.float64]
    n_transitions: int
    is_ergodic: bool | None
    backend: str
    states: tuple[str, ...] = field(default_factory=lambda: ('unbiased', 'biased'))


def transition_counts(
    seq_iter: Iterable[Sequence[int]],
    n_states: int = 2,
) -> tuple[NDArray[np.float64], int]:
    """Tabulate per-step transition counts from state sequences.

    Args:
        seq_iter: Iterable of integer-coded state sequences. Sequences of
            length ``< 2`` contribute no transitions.
        n_states: Cardinality of the state space; sets the matrix shape.

    Returns:
        ``(counts, n_transitions)`` where ``counts[i, j]`` is the observed
        count of transitions from state ``i`` to state ``j`` and
        ``n_transitions`` is the total across all sequences.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    n_tr = 0
    for raw in seq_iter:
        seq = np.asarray(raw, dtype=int)
        if seq.size < 2:
            continue
        np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
        n_tr += int(seq.size - 1)
    return counts, n_tr


def _normalise_rows(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-stochastic-ise; rows that sum to zero become uniform 1/N rows."""
    n_states = matrix.shape[1]
    row_sums = matrix.sum(axis=1, keepdims=True)
    uniform_row = np.full_like(matrix, 1.0 / n_states)
    safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalised = np.where(row_sums == 0, uniform_row, matrix / safe_row_sums)
    return np.asarray(normalised, dtype=np.float64)


def _stationary_via_quantecon(
    p_norm: NDArray[np.float64],
    states: tuple[str, ...],
) -> tuple[NDArray[np.float64], bool]:
    mc = qe.MarkovChain(p_norm, state_values=list(states))
    sd = np.asarray(mc.stationary_distributions, dtype=np.float64)
    pi: NDArray[np.float64] = (
        np.asarray(sd[0], dtype=np.float64)
        if sd.size
        else np.full(p_norm.shape[0], 1.0 / p_norm.shape[0], dtype=np.float64)
    )
    return pi, bool(mc.is_irreducible)


def _stationary_via_numpy(p_norm: NDArray[np.float64]) -> NDArray[np.float64]:
    eigvals, eigvecs = np.linalg.eig(p_norm.T)
    pi_raw = np.asarray(
        np.real(eigvecs[:, int(np.argmin(np.abs(eigvals - 1.0)))]), dtype=np.float64
    )
    total = pi_raw.sum()
    if total == 0:
        return np.full(p_norm.shape[0], 1.0 / p_norm.shape[0], dtype=np.float64)
    return np.asarray(pi_raw / total, dtype=np.float64)


def markov_chain_summary(
    transition_matrix: NDArray[np.float64],
    n_transitions: int = 0,
    states: Sequence[str] = ('unbiased', 'biased'),
) -> MarkovSummary:
    """Estimate the stationary distribution and ergodicity of a Markov chain.

    Args:
        transition_matrix: Row-stochastic transition matrix; rows that sum
            to zero are re-normalised to a uniform distribution to keep
            the result well-defined.
        n_transitions: Number of observed transitions backing the matrix;
            recorded into the result but not otherwise used.
        states: Labels for the rows / columns of the transition matrix.

    Returns:
        :class:`MarkovSummary`. ``is_ergodic`` is ``None`` when the
        ``quantecon`` backend is unavailable.
    """
    p_norm = _normalise_rows(np.asarray(transition_matrix, dtype=float))
    state_tuple = tuple(states)
    if _HAS_QE:
        pi, is_ergodic = _stationary_via_quantecon(p_norm, state_tuple)
        return MarkovSummary(
            P=p_norm,
            pi=pi,
            n_transitions=int(n_transitions),
            is_ergodic=is_ergodic,
            backend=_QE_BACKEND_NAME,
            states=state_tuple,
        )
    return MarkovSummary(
        P=p_norm,
        pi=_stationary_via_numpy(p_norm),
        n_transitions=int(n_transitions),
        is_ergodic=None,
        backend=_QE_BACKEND_NAME,
        states=state_tuple,
    )


def permutation_pvalue(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    statistic: Callable[..., float],
    *,
    n_resamples: int = 4000,
    seed: int = 42,
    permutation_type: str = 'independent',
    alternative: str = 'two-sided',
) -> float:
    """Compute a permutation p-value for an arbitrary statistic.

    Args:
        sample_a: First sample.
        sample_b: Second sample.
        statistic: Callable ``f(a, b) -> float`` consumed by
            :func:`scipy.stats.permutation_test`.
        n_resamples: Number of random permutations.
        seed: PRNG seed for reproducibility.
        permutation_type: One of ``'independent'``, ``'samples'``,
            ``'pairings'`` (see SciPy docs).
        alternative: One of ``'two-sided'``, ``'less'``, ``'greater'``.

    Returns:
        The permutation p-value, with the standard
        ``(observed + 1) / (n_resamples + 1)`` continuity correction.
    """
    res = st.permutation_test(
        (np.asarray(sample_a), np.asarray(sample_b)),
        statistic=statistic,
        permutation_type=permutation_type,
        n_resamples=n_resamples,
        vectorized=False,
        alternative=alternative,
        random_state=seed,
    )
    return float(res.pvalue)


def anova_two_way(df: pd.DataFrame, dv: str, factors: list[str]) -> pd.DataFrame:
    r"""Run a Type-II two-way ANOVA in long format.

    Args:
        df: Long-format dataframe with one row per observation.
        dv: Name of the dependent-variable column.
        factors: Two between-subject factor column names.

    Returns:
        A tidy ANOVA table from :func:`pingouin.anova` with F, p-value,
        and partial-:math:`\eta^2`. Returns an empty dataframe when any
        required column is missing.
    """
    if df.empty or dv not in df.columns or any(f not in df.columns for f in factors):
        return pd.DataFrame()
    return pg.anova(data=df, dv=dv, between=factors, ss_type=2, detailed=True)


@dataclass(frozen=True, slots=True)
class CIResult:
    """Frozen bootstrap-CI result with explicit success flag.

    Attributes:
        mean: Sample mean of the cleaned input.
        lo: Lower bound of the confidence interval.
        hi: Upper bound of the confidence interval.
        ok: ``False`` when the CI could not be computed (e.g. degenerate
            input); callers should branch on this flag instead of catching
            exceptions.
        reason: Human-readable explanation populated when ``ok`` is
            ``False`` or the CI is degenerate.
    """

    mean: float
    lo: float
    hi: float
    ok: bool
    reason: str = ''


@dataclass(frozen=True, slots=True)
class PairedDiffCI:
    """Bootstrap CI for the mean paired difference ``a - b``.

    Attributes:
        mean: Sample mean of ``a - b``.
        ci_lo: Lower percentile bound.
        ci_hi: Upper percentile bound.
        n: Number of paired observations.
    """

    mean: float
    ci_lo: float
    ci_hi: float
    n: int


def safe_wilcoxon_p(
    arm_a: NDArray[np.float64],
    arm_b: NDArray[np.float64],
    *,
    zero_method: str = 'wilcox',
) -> float:
    """Compute a paired Wilcoxon signed-rank p-value with a degenerate-input fallback.

    Args:
        arm_a: First paired sample.
        arm_b: Second paired sample; must have the same length as ``arm_a``.
        zero_method: How :func:`scipy.stats.wilcoxon` handles zero
            differences (``'wilcox'``, ``'pratt'``, or ``'zsplit'``).

    Returns:
        The two-sided p-value, or ``nan`` when fewer than six paired,
        non-zero, non-NaN differences are available.
    """
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.size != b.size or a.size < 6:
        return float('nan')
    diff = a - b
    diff = diff[~np.isnan(diff)]
    if diff.size < 6 or np.all(diff == 0):
        return float('nan')
    return float(st.wilcoxon(a, b, zero_method=zero_method).pvalue)


def safe_bootstrap_mean_ci(
    diff: NDArray[np.float64],
    *,
    n_resamples: int = 4000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CIResult:
    """Compute a percentile bootstrap CI of the mean.

    Args:
        diff: 1-D array of (typically paired) differences. ``NaN`` entries
            are dropped.
        n_resamples: Number of bootstrap resamples.
        alpha: Two-sided significance level; the CI has nominal coverage
            ``1 - alpha``.
        seed: PRNG seed for reproducibility.

    Returns:
        :class:`CIResult`. ``ok`` is ``False`` when fewer than six
        non-``NaN`` values remain. A constant input returns ``ok=True``
        with a degenerate ``lo == mean == hi`` CI and a populated
        ``reason``.
    """
    cleaned = np.asarray(diff, dtype=float)
    cleaned = cleaned[~np.isnan(cleaned)]
    if cleaned.size < 6:
        return CIResult(
            mean=float('nan'),
            lo=float('nan'),
            hi=float('nan'),
            ok=False,
            reason=f'n={cleaned.size} < 6',
        )
    if np.all(cleaned == cleaned[0]):
        m = float(cleaned[0])
        return CIResult(mean=m, lo=m, hi=m, ok=True, reason='constant sample (degenerate CI)')
    res = st.bootstrap(
        (cleaned,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        method='percentile',
        random_state=seed,
    )
    return CIResult(
        mean=float(cleaned.mean()),
        lo=float(res.confidence_interval.low),
        hi=float(res.confidence_interval.high),
        ok=True,
    )


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR (Benjamini & Hochberg, 1995, JRSSB 57:289-300).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BHFDRResult:
    """Result of Benjamini-Hochberg FDR adjustment.

    Attributes:
        q_values: Adjusted q-values, one per input p-value (input order).
        rejected: Boolean rejection mask at level ``alpha`` (input order).
        alpha: The FDR level used.
    """

    q_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float


def bh_fdr(p_values: Sequence[float], *, alpha: float = 0.05) -> BHFDRResult:
    """Benjamini-Hochberg step-up FDR adjustment.

    Args:
        p_values: Two-sided p-values (each in ``[0, 1]``).
        alpha: Target false-discovery rate (``0 < alpha < 1``).

    Returns:
        A :class:`BHFDRResult` whose ``q_values`` and ``rejected`` are
        aligned to the input ordering.

    Raises:
        ValueError: If ``p_values`` is empty, contains values outside
            ``[0, 1]``, or ``alpha`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha must lie in (0, 1); got {alpha}')
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        raise ValueError('p_values must be non-empty')
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError('p_values must lie in [0, 1] and be finite')
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate((ranked * n / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    rejected_sorted = q_sorted <= alpha
    q_out = np.empty_like(q_sorted)
    rej_out = np.empty_like(rejected_sorted)
    q_out[order] = q_sorted
    rej_out[order] = rejected_sorted
    return BHFDRResult(
        q_values=tuple(float(x) for x in q_out),
        rejected=tuple(bool(x) for x in rej_out),
        alpha=alpha,
    )


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised paired-difference effect size (Cohen's d on differences).

    Args:
        a: First measurement vector.
        b: Second measurement vector aligned with ``a`` (same length).

    Returns:
        ``mean(a - b) / sd(a - b)`` with sample (ddof=1) standard deviation;
        ``nan`` if the difference is constant or shorter than 2.

    Raises:
        ValueError: If ``a`` and ``b`` differ in length.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f'cohens_d_paired: shape mismatch {arr_a.shape} vs {arr_b.shape}')
    diff = arr_a - arr_b
    if diff.size < 2:
        return float('nan')
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return float('nan')
    return float(np.mean(diff) / sd)


def hedges_g_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Hedges' g (small-sample bias-corrected paired Cohen's d)."""
    d = cohens_d_paired(a, b)
    n = len(a)
    if n < 2 or not np.isfinite(d):
        return float('nan')
    df = n - 1
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(j * d)


def paired_diff_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> PairedDiffCI:
    """Percentile bootstrap CI for the mean paired difference ``a - b``.

    Args:
        a: First measurement vector.
        b: Second measurement vector aligned with ``a``.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Two-sided confidence level in ``(0, 1)``.
        seed: RNG seed; ``None`` for non-deterministic.

    Returns:
        :class:`CIResult` with ``mean = mean(a - b)``, percentile bounds,
        and ``n = len(a)``.

    Raises:
        ValueError: If lengths differ or ``confidence`` is out of range.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f'paired_diff_bootstrap_ci: shape mismatch {arr_a.shape} vs {arr_b.shape}',
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f'confidence must be in (0, 1); got {confidence}')
    diff = arr_a - arr_b
    n = diff.size
    if n == 0:
        return PairedDiffCI(mean=float('nan'), ci_lo=float('nan'), ci_hi=float('nan'), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return PairedDiffCI(mean=float(diff.mean()), ci_lo=float(lo), ci_hi=float(hi), n=n)
