"""Bias-lifecycle metrics derived from per-turn ``stream_round_metrics.csv``.

Operationalises self-correction as a single concept: a sample is *introduced*
to bias if any turn shows ``bias_prevalence > 0``; it is *recovered* if it was
introduced AND the final turn is clean. The first turn at which
``bias_prevalence`` returns (and stays) at zero is the *recovery turn*. The
three resulting lifecycle states ``A_never_biased / B_recovered / C_persistent``
fold the legacy four-state taxonomy (``B_resolved_early``, ``C_resolved_at_end``)
into a single ``B_recovered`` bucket because the user asked us to treat
"resolved at end" as a special case of self-correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import Final

import numpy as np
import polars as pl

__all__ = [
    'LIFECYCLE_STATES',
    'LifecycleAggregate',
    'aggregate_lifecycle',
    'derive_lifecycle',
]

LIFECYCLE_STATES: Final[tuple[str, str, str]] = (
    'A_never_biased',
    'B_recovered',
    'C_persistent',
)


@dataclass(frozen=True, slots=True)
class LifecycleAggregate:
    """Per-arm lifecycle aggregate.

    Attributes:
        n_samples: Total samples.
        state_counts: Mapping ``state -> count``.
        state_proportions: Mapping ``state -> proportion``.
        recovery_rate: ``B_recovered / (B_recovered + C_persistent)`` (NaN if no introductions).
        introduction_rate: ``(B_recovered + C_persistent) / n_samples``.
        median_recovery_turn: Median recovery turn over recovered samples (NaN if none).
        recovery_turn_counts: Mapping ``turn_index -> count`` for recovered samples.
    """

    n_samples: int
    state_counts: dict[str, int]
    state_proportions: dict[str, float]
    recovery_rate: float
    introduction_rate: float
    median_recovery_turn: float
    recovery_turn_counts: dict[int, int]


def derive_lifecycle(round_df: pl.DataFrame) -> pl.DataFrame:
    """Derive per-sample lifecycle state from per-turn ``bias_prevalence``.

    Pure-declarative polars pipeline: a sort + reverse-cumulative-sum on the
    binarised series gives, for each turn, the count of *future* biased
    turns; the recovery turn of a sample is the smallest ``turn_index``
    whose ``_future_biased == 0`` and whose preceding turns saw at least
    one biased turn. The whole derivation is one ``group_by(sample_id).agg``.

    Args:
        round_df: Per-turn dataframe with at least the columns
            ``sample_id``, ``turn_index``, ``bias_prevalence``.

    Returns:
        A polars DataFrame with one row per ``sample_id`` and columns:
        ``sample_id``, ``n_turns``, ``ever_biased`` (bool),
        ``end_biased`` (bool), ``is_introduced`` (bool),
        ``is_recovered`` (bool), ``recovery_turn`` (Int64, null if not
        recovered or never biased), ``lifecycle_state`` (str).

    Raises:
        ValueError: If a required column is missing or ``round_df`` is empty.
    """
    required = {'sample_id', 'turn_index', 'bias_prevalence'}
    missing = required - set(round_df.columns)
    if missing:
        raise ValueError(f'derive_lifecycle requires columns {required}; missing {missing}')
    if round_df.height == 0:
        raise ValueError('derive_lifecycle requires non-empty round_df')

    biased = (pl.col('bias_prevalence') > 0.0).cast(pl.Int64)

    # Compute per-turn prefix and suffix counts of biased turns within each
    # sample, then aggregate per sample.  Suffix is total - prefix + current
    # (so suffix at turn t == 0 iff no biased turn from t onward).
    enriched = (
        round_df
        .lazy()
        .sort(['sample_id', 'turn_index'])
        .with_columns(biased.alias('_b'))
        .with_columns(
            pl.col('_b').cum_sum().over('sample_id').alias('_prefix_b'),
            pl.col('_b').sum().over('sample_id').alias('_total_b'),
        )
        .with_columns(
            (pl.col('_total_b') - pl.col('_prefix_b') + pl.col('_b')).alias('_suffix_b'),
        )
    )

    per_sample = (
        enriched
        .group_by('sample_id', maintain_order=True)
        .agg(
            pl.len().alias('n_turns'),
            (pl.col('_total_b') > 0).first().alias('ever_biased'),
            (pl.col('_b').last() > 0).alias('end_biased'),
            pl.col('turn_index').alias('_turns'),
            pl.col('_suffix_b').alias('_suffix_seq'),
            pl.col('_prefix_b').alias('_prefix_seq'),
        )
        .collect()
    )

    # Vectorised recovery_turn: first turn whose strict suffix is zero AND
    # prefix is positive.  Using zip avoids a Python-side group_by loop.
    def _first_recovery(turns: list[int], suffix: list[int], prefix: list[int]) -> int | None:
        for t, s, p in zip(turns, suffix, prefix, strict=True):
            if s == 0 and p > 0:
                return int(t)
        return None

    rec_turns = list(
        starmap(
            _first_recovery,
            zip(
                per_sample.get_column('_turns').to_list(),
                per_sample.get_column('_suffix_seq').to_list(),
                per_sample.get_column('_prefix_seq').to_list(),
                strict=True,
            ),
        )
    )

    return (
        per_sample
        .with_columns(
            pl.Series('recovery_turn', rec_turns, dtype=pl.Int64),
            (pl.col('ever_biased')).alias('is_introduced'),
            (pl.col('ever_biased') & ~pl.col('end_biased')).alias('is_recovered'),
        )
        .with_columns(
            pl
            .when(~pl.col('ever_biased'))
            .then(pl.lit('A_never_biased'))
            .when(pl.col('is_recovered'))
            .then(pl.lit('B_recovered'))
            .otherwise(pl.lit('C_persistent'))
            .alias('lifecycle_state'),
        )
        .select(
            'sample_id',
            'n_turns',
            'ever_biased',
            'end_biased',
            'is_introduced',
            'is_recovered',
            'recovery_turn',
            'lifecycle_state',
        )
    )


def aggregate_lifecycle(lifecycle_df: pl.DataFrame) -> LifecycleAggregate:
    """Aggregate a per-sample lifecycle DataFrame into arm-level summaries.

    Args:
        lifecycle_df: Output of :func:`derive_lifecycle`.

    Returns:
        :class:`LifecycleAggregate`.

    Raises:
        ValueError: If ``lifecycle_df`` is empty or missing
            ``lifecycle_state`` / ``recovery_turn``.
    """
    required = {'lifecycle_state', 'recovery_turn'}
    missing = required - set(lifecycle_df.columns)
    if missing:
        raise ValueError(f'aggregate_lifecycle requires columns {required}; missing {missing}')
    n = lifecycle_df.height
    if n == 0:
        raise ValueError('aggregate_lifecycle requires non-empty input')

    state_counts: dict[str, int] = dict.fromkeys(LIFECYCLE_STATES, 0)
    for s in lifecycle_df.get_column('lifecycle_state').to_list():
        state_counts[str(s)] += 1
    state_proportions = {s: state_counts[s] / n for s in LIFECYCLE_STATES}

    intros = state_counts['B_recovered'] + state_counts['C_persistent']
    recovery_rate = (state_counts['B_recovered'] / intros) if intros > 0 else float('nan')
    introduction_rate = intros / n

    rt = (
        lifecycle_df
        .filter(pl.col('lifecycle_state') == 'B_recovered')
        .get_column('recovery_turn')
        .drop_nulls()
        .to_list()
    )
    median_recovery_turn = float(np.median(rt)) if rt else float('nan')
    rt_counts: dict[int, int] = {}
    for t in rt:
        rt_counts[int(t)] = rt_counts.get(int(t), 0) + 1

    return LifecycleAggregate(
        n_samples=n,
        state_counts=state_counts,
        state_proportions=state_proportions,
        recovery_rate=recovery_rate,
        introduction_rate=introduction_rate,
        median_recovery_turn=median_recovery_turn,
        recovery_turn_counts=dict(sorted(rt_counts.items())),
    )


def _ensure_sequence(x: Sequence[object]) -> Sequence[object]:
    """Internal helper to force list-like for downstream typing."""
    return list(x)
