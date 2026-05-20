"""Adapter that exposes a DSPy MAS module as an MLflow ``predict_fn``.

The adapter is intentionally thin: it forwards the named BBQ-style
inputs to the wrapped :class:`dspy.Module` and runs the program's
output through :class:`MASPredictionEnvelope` so any shape / consistency
problem surfaces as a :class:`pydantic.ValidationError` with field
paths instead of a generic ``RuntimeError``.

All validation logic lives on
:class:`bias_mitigation.mas.prediction_models.MASPredictionEnvelope` —
this module only translates between the prediction object DSPy returns
and the dict shape ``model_validate`` expects, then rebuilds a
``dspy.Prediction`` with the normalised history / final-answers
payload that downstream scorers consume.
"""

from typing import Any

import dspy

from ..prediction_models import MASPredictionEnvelope

_MAS_INPUT_FIELDS = ('context', 'question', 'ans0', 'ans1', 'ans2', 'category')


class PredictFnAdapter:
    """Adapter that exposes a DSPy module as an MLflow ``predict_fn``."""

    def __init__(self, program: dspy.Module):
        self.program = program

    @staticmethod
    def _require_mapping(prediction: dspy.Prediction, field_name: str) -> dict[str, Any]:
        """Pull ``field_name`` off ``prediction`` and assert it is a dict."""
        value = getattr(prediction, field_name, None)
        if not isinstance(value, dict):
            raise TypeError(
                f'MASProgram prediction missing valid {field_name} mapping: {prediction!r}',
            )
        return value

    @classmethod
    def _normalize_prediction(cls, prediction: dspy.Prediction) -> dspy.Prediction:
        """Validate the raw prediction via Pydantic and rebuild it normalized.

        ``MASPredictionEnvelope.model_validate`` does all the work:
        accepts ``dspy.Prediction``-shaped turns or plain dicts, defaults
        missing ``agent_name`` to the parent history key, and enforces
        cross-field invariants (non-empty history, matching keys
        between ``history`` and ``final_answers``).
        """
        envelope = MASPredictionEnvelope.model_validate({
            'history': cls._require_mapping(prediction, 'history'),
            'final_answers': {'values': cls._require_mapping(prediction, 'final_answers')},
        })
        normalized_history = {
            agent_name: [
                dspy.Prediction(
                    answer=turn.answer,
                    reasoning=turn.reasoning,
                    agent_name=turn.agent_name or agent_name,
                )
                for turn in agent_history.turns
            ]
            for agent_name, agent_history in envelope.history.items()
        }
        payload: dict[str, Any] = {
            'history': normalized_history,
            'final_answers': dict(envelope.final_answers.values),
        }
        for optional_key in ('sample_id', 'sample_run_id'):
            optional_value = getattr(prediction, optional_key, None)
            if optional_value is not None:
                payload[optional_key] = str(optional_value)
        return dspy.Prediction(**payload)

    def __call__(self, **inputs: Any) -> dspy.Prediction:
        """Forward inputs to ``program`` and validate the prediction shape."""
        forwarded = {
            **{name: inputs[name] for name in _MAS_INPUT_FIELDS},
            'stereotyped_groups': inputs.get('stereotyped_groups'),
            **{k: v for k, v in inputs.items() if k not in {*_MAS_INPUT_FIELDS, 'stereotyped_groups'}},
        }
        prediction = self.program(**forwarded)
        if not isinstance(prediction, dspy.Prediction):
            raise TypeError(f'MASProgram returned invalid prediction object: {prediction!r}')
        return self._normalize_prediction(prediction)
