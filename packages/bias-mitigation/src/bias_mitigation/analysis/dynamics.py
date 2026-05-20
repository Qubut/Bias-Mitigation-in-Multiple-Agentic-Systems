"""Two-state Markov dynamics for the per-turn ``bias_prevalence > 0`` indicator.

Fits an empirical 2x2 transition matrix per arm (or per group) from the
per-turn time series and exposes the standard ergodic-chain quantities
(stationary distribution, recovery probability, expected recovery time,
geometric pmf of recovery turn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = ['TwoStateMarkov', 'fit_two_state_markov']


@dataclass(frozen=True)
class TwoStateMarkov:
    """Result of fitting a 2-state Markov chain to per-turn bias indicators.

    Attributes:
        P: 2x2 row-stochastic transition matrix (``P[i, j] = P(s_{t+1}=j | s_t=i)``).
        n_transitions: Total number of (from, to) transitions used.
        n_samples: Number of distinct sample-id sequences used.
        p_persist: ``P[1, 1]`` — probability biased state persists.
        p_emerge:  ``P[0, 1]`` — probability of emerging into the biased state.
        p_recover: ``P[1, 0]`` — probability of recovery from the biased state.
        pi_biased: Stationary probability of the biased state.
    """

    P: np.ndarray
    n_transitions: int
    n_samples: int

    @property
    def p_persist(self) -> float:
        return float(self.P[1, 1])

    @property
    def p_emerge(self) -> float:
        return float(self.P[0, 1])

    @property
    def p_recover(self) -> float:
        return float(self.P[1, 0])

    @property
    def pi_biased(self) -> float:
        denom = self.p_emerge + self.p_recover
        if denom <= 0.0:
            return float('nan')
        return float(self.p_emerge / denom)

    def expected_recovery_turn(self) -> float:
        """Expected number of turns until recovery, conditional on being biased.

        For a geometric distribution with success probability ``p_recover``,
        the mean is ``1 / p_recover``.
        """
        p = self.p_recover
        return float('inf') if p <= 0.0 else 1.0 / p

    def geometric_recovery_pmf(self, turns: int = 7) -> dict[int, float]:
        """Return ``{t: P(T_recover = t | biased)}`` for ``t = 1..turns``.

        ``P(T = t) = (1 - p_recover) ** (t - 1) * p_recover``.
        """
        if turns < 1:
            raise ValueError(f'turns must be >= 1; got {turns}')
        p = self.p_recover
        return {t: (1.0 - p) ** (t - 1) * p for t in range(1, turns + 1)}


def fit_two_state_markov(
    round_df: pl.DataFrame,
    *,
    state_col: str = 'bias_prevalence',
    sample_col: str = 'sample_id',
    turn_col: str = 'turn_index',
    laplace: float = 1.0,
) -> TwoStateMarkov:
    """Fit a 2-state Markov chain on the ``bias_prevalence > 0`` indicator.

    Args:
        round_df: Per-turn dataframe with at least ``sample_col``,
            ``turn_col`` and ``state_col``.
        state_col: Column carrying the per-turn bias prevalence; the state
            is binarised as ``state_col > 0``.
        sample_col: Column identifying a per-sample dialogue (turn order
            is reset within each sample).
        turn_col: Column carrying the per-turn index, used to sort.
        laplace: Add-one (or smaller) smoothing constant added to every
            transition count so each row remains a valid probability
            distribution even when one outgoing edge is unobserved.

    Returns:
        :class:`TwoStateMarkov` with the row-stochastic matrix and counts.

    Raises:
        ValueError: If the required columns are missing or ``round_df`` is empty.
    """
    required = {sample_col, turn_col, state_col}
    missing = required - set(round_df.columns)
    if missing:
        raise ValueError(f'fit_two_state_markov: missing columns {sorted(missing)}')
    if round_df.is_empty():
        raise ValueError('fit_two_state_markov: round_df is empty')
    if laplace < 0.0:
        raise ValueError(f'laplace must be >= 0; got {laplace}')

    # Single-pass declarative pipeline:
    #   sort -> binarise -> within-sample lag -> drop first row per sample
    #   -> group-and-count the four (prev, curr) cells.
    transitions = (
        round_df
        .lazy()
        .sort([sample_col, turn_col])
        .with_columns((pl.col(state_col) > 0).cast(pl.Int8).alias('_state'))
        .with_columns(pl.col('_state').shift(1).over(sample_col).alias('_prev'))
        .filter(pl.col('_prev').is_not_null())
        .group_by(['_prev', '_state'])
        .len()
        .collect()
    )

    counts = np.zeros((2, 2), dtype=np.int64)
    for row in transitions.iter_rows(named=True):
        counts[int(row['_prev']), int(row['_state'])] = int(row['len'])
    n_transitions = int(counts.sum())
    n_samples = int(round_df.get_column(sample_col).n_unique())

    smoothed = counts.astype(np.float64) + laplace
    row_sums = smoothed.sum(axis=1, keepdims=True)
    # Defensive: with laplace=0 a row sum can be zero (no transitions out of
    # an unobserved state); fall back to a uniform 1/k distribution so the
    # matrix remains row-stochastic.
    safe_rows = np.where(row_sums > 0.0, row_sums, 1.0)
    P = np.where(row_sums > 0.0, smoothed / safe_rows, 1.0 / smoothed.shape[1])  # noqa: N806
    return TwoStateMarkov(P=P, n_transitions=n_transitions, n_samples=n_samples)
