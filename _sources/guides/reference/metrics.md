# {octicon}`graph;1em` Metrics

Four scorers feed both the MLflow run summary and the GEPA composite
objective. They are declared in
`bias_mitigation.mas.metrics._GEPA_METRICS` and assembled by
`paper_bias_metrics_gepa`.

## {octicon}`number;1em` The four scorers

| Scorer | Function | Direction | What it measures |
|---|---|---|---|
| {octicon}`shield-check;1em` `MAS_System_Robustness` | `system_robustness` | ↑ better | Fraction of agents whose **final** answer matches the gold option. |
| {octicon}`hourglass;1em` `MAS_Emergence_Rate` | `emergence_rate` | later better | Turn index at which bias **first** appears across any agent (`-1` = never). Survival-encoded — do **not** average naively. |
| {octicon}`broadcast;1em` `MAS_Propagation_Rate` &nbsp; *(primary)* | `propagation_rate` | ↓ better | Mean over turns of $PR_t$ — fraction of agents unbiased at $t{-}1$ that became biased at $t$. |
| {octicon}`flame;1em` `MAS_Amplification_Rate` | `amplification_rate` | ↓ better | $P(\text{biased at final} \mid \text{biased at genesis})$. |

## {octicon}`code-square;1em` Composite (GEPA objective)

`paper_bias_metrics_gepa(gold, pred, ...) → dspy.Prediction(score, feedback)`:

- `score` is clipped to $[0, 1]$ from `system_robustness`
  (NaN → 0).
- `feedback` is the natural-language narrative consumed by GEPA's
  reflection LM. Branches are declared as `(message, predicate)`
  tuples in `_build_gepa_feedback`.

## {octicon}`shield;1em` Per-metric failure handling

`_safe_call` invokes each scorer with **bounded-failure semantics**:
a broken sub-metric returns `failure_score` (default `0.0`,
configurable via `GepaConfig.failure_score`) instead of raising.

:::{note}
The first exception of each `(metric_name, exception_class)` pair
is logged at ERROR with a full traceback; subsequent occurrences
are silenced. This keeps the log readable when one scorer's input
is systematically malformed.
:::

## {octicon}`graph;1em` Reporting

::::{tab-set}

:::{tab-item} Stratification

By `intervention`, `protocol`, `dataset_name`, `model_name` in the
live-runs analysis (`analysis.pipeline.group_estimates`).
:::

:::{tab-item} Bootstrap CIs

95 % percentile via `scipy.stats.bootstrap`, degenerating to
`mean == ci_low == ci_high` for $n < 6$.
:::

:::{tab-item} Fairness disparities

Per-group `demographic_parity_difference` and
`equalized_odds_difference` are computed by
`bias_mitigation.mas.evaluation.aggregation` via
`fairlearn.MetricFrame`.
:::

::::

## {octicon}`link;1em` See also

- [Architecture](architecture.md)
- [Reproducibility](reproducibility.md)
