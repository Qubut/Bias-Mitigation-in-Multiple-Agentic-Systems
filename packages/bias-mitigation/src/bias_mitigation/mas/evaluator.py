from typing import Any

import dspy

from bias_mitigation.mas.metrics import paper_bias_metrics


class MASEvaluator:
    """Enterprise declarative evaluator - zero imperative code."""

    def __init__(self, devset: list[dspy.Example]):
        self.evaluator = dspy.Evaluate(
            devset=devset,
            metric=paper_bias_metrics,
            num_threads=8,
            display_progress=True,
            display_table=5,
        )

    def __call__(self, program: dspy.Module, devset: list[dspy.Example]) -> dict[str, Any]:
        """One call → all paper metrics aggregated."""
        score, results = self.evaluator(program, devset=devset)
        return {
            'system_robustness': score,
            'detailed_results': results,  # contains per-example emergence/PR_t/etc.
            'config': getattr(program, 'config', None),
        }
