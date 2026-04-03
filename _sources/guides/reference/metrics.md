# Metrics Definitions

This page defines the primary and secondary outcomes used in the study.

## Primary Outcome

### Propagation Rate ($PR_t$)

Measures the extent to which biased outputs propagate across agents over turns.
Reported as an aggregated run-level metric over interaction rounds.

## Secondary Outcomes

### Emergence Rate ($ER_t$)

Captures when bias first appears in the conversation trajectory.
Used to profile onset timing across intervention conditions.

### Amplification Rate ($AR_t$)

Measures growth of bias relative to initial biased signals.
Used to detect whether interaction dynamics escalate or dampen bias.

### System Robustness

Run-level robustness score derived from final answer behavior and bias metrics.
Used as a summary indicator alongside $PR_t$, $ER_t$, and $AR_t$.

## Reporting Guidance

- Report metrics per intervention condition and protocol.
- Keep config and data split controls fixed for fair comparisons.
- Report exclusions and sensitivity analyses when technical failures occur.

## Related Pages

- {doc}`/guides/reference/index`
- {doc}`/guides/reference/reproducibility`
- {doc}`/api/index`
