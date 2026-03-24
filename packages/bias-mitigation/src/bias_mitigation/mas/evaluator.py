from typing import Any

import dspy

from bias_mitigation.mas.metrics import paper_bias_metrics


class MASEvaluator:
    """
    Evaluates a Multiple Agentic System (MAS) program against bias metrics.

    This evaluator uses the `dspy` evaluation utility to run a given program
    across a development dataset, calculating system robustness and tracking
    detailed per-example results based on predefined paper metrics.

    Args:
        devset (list[dspy.Example]): The development dataset used to initialize
            the evaluation environment.
    """
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
