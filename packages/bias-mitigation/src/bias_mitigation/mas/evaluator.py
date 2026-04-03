"""Evaluation helpers that bridge MAS programs with MLflow GenAI scoring."""

from typing import Any

import dspy
import mlflow
from mlflow.genai import evaluate

from .metrics import (
    amplification_rate,
    emergence_rate,
    propagation_rate,
    system_robustness,
)


class PredictFnAdapter:
    """Adapter that exposes a DSPy module as an MLflow ``predict_fn``."""

    def __init__(self, program: dspy.Module):
        self.program = program

    @mlflow.trace(name='PredictFnAdapter_call', span_type=mlflow.entities.SpanType.AGENT)
    def __call__(self, **inputs: dict[str, Any]) -> dspy.Prediction:
        """Forward a normalized input mapping to ``program.forward``."""
        # Forward only the fields the MASProgram expects
        return self.program(
            context=inputs['context'],
            question=inputs['question'],
            ans0=inputs['ans0'],
            ans1=inputs['ans1'],
            ans2=inputs['ans2'],
            category=inputs['category'],
            stereotyped_groups=inputs.get('stereotyped_groups'),
            **{
                k: v
                for k, v in inputs.items()
                if k
                not in {
                    'context',
                    'question',
                    'ans0',
                    'ans1',
                    'ans2',
                    'category',
                    'stereotyped_groups',
                }
            },
        )


class MASEvaluator:
    """Run GenAI evaluation over a MAS program using paper-aligned scorers."""

    def __init__(self, devset: list[dspy.Example]):
        self.devset = devset

    def _to_mlflow_eval_dataset(self, devset: list[dspy.Example]) -> list[dict[str, Any]]:
        """Convert DSPy examples into MLflow GenAI dataset records."""
        return [{'inputs': example.toDict()} for example in devset]

    def __call__(
        self, program: dspy.Module, devset: list[dspy.Example] | None = None
    ) -> dict[str, Any]:
        """Evaluate ``program`` and return summary and raw GenAI results.

        Returns:
            Dictionary with aggregate robustness, full GenAI output object,
            and metric map emitted by MLflow evaluation.
        """
        devset = devset or self.devset
        eval_data = self._to_mlflow_eval_dataset(devset)
        predict_fn = PredictFnAdapter(program)
        genai_result = evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[
                system_robustness,
                emergence_rate,
                amplification_rate,
                propagation_rate,
            ],
        )

        return {
            'system_robustness': genai_result.metrics.get('MAS_Bias_Metrics', 0.0),
            'genai_evaluation': genai_result,
            'detailed_results': None,
            'config': getattr(program, 'config', None),
            'genai_metrics': genai_result.metrics,
        }
