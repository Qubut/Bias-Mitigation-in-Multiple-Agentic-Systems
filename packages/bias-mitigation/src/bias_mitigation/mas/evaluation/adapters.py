from typing import Any

import dspy
from returns.functions import raise_exception
from returns.result import Result, safe

from ..prediction_models import AgentHistory, AgentTurn, FinalAnswers, MASPredictionEnvelope


class PredictFnAdapter:
    """Adapter that exposes a DSPy module as an MLflow ``predict_fn``."""

    def __init__(self, program: dspy.Module):
        self.program = program

    @staticmethod
    def _coerce_turn(turn: Any, agent_name: str) -> AgentTurn:
        """Convert one raw turn object into validated ``AgentTurn`` model."""
        match turn:
            case dspy.Prediction():
                payload = {
                    'answer': getattr(turn, 'answer', None),
                    'reasoning': getattr(turn, 'reasoning', None),
                    'agent_name': getattr(turn, 'agent_name', agent_name),
                }
            case dict():
                payload = {
                    'answer': turn.get('answer'),
                    'reasoning': turn.get('reasoning'),
                    'agent_name': turn.get('agent_name', agent_name),
                }
            case _:
                raise TypeError(f'Invalid turn type in history for {agent_name}: {type(turn)!r}')
        return AgentTurn.model_validate(payload)

    @staticmethod
    def _require_mapping_field(prediction: dspy.Prediction, field_name: str) -> dict[str, Any]:
        value = getattr(prediction, field_name, None)
        if not isinstance(value, dict):
            raise TypeError(
                f'MASProgram prediction missing valid {field_name} mapping: {prediction!r}'
            )
        return value

    @staticmethod
    def _validate_envelope(envelope: MASPredictionEnvelope) -> None:
        if not envelope.history:
            raise ValueError('Prediction history is empty; cannot score MAS metrics.')

        for agent_name, agent_history in envelope.history.items():
            if not agent_history.turns:
                raise ValueError(f'Prediction history for {agent_name} has no turns.')

        history_agents = set(envelope.history.keys())
        final_answer_agents = set(envelope.final_answers.values.keys())
        if history_agents != final_answer_agents:
            raise ValueError(
                'Mismatch between history agent keys and final_answers keys: '
                f'history={sorted(history_agents)}, final_answers={sorted(final_answer_agents)}'
            )

    @staticmethod
    def _build_normalized_payload(
        prediction: dspy.Prediction,
        envelope: MASPredictionEnvelope,
    ) -> dict[str, Any]:
        normalized_history = {
            agent_name: [
                dspy.Prediction(
                    answer=turn.answer,
                    reasoning=turn.reasoning,
                    agent_name=turn.agent_name or agent_name,
                )
                for turn in turns.turns
            ]
            for agent_name, turns in envelope.history.items()
        }

        normalized_payload: dict[str, Any] = {
            'history': normalized_history,
            'final_answers': dict(envelope.final_answers.values),
        }

        for optional_key in ('sample_id', 'sample_run_id'):
            optional_value = getattr(prediction, optional_key, None)
            if optional_value is not None:
                normalized_payload[optional_key] = str(optional_value)

        return normalized_payload

    @classmethod
    def _normalize_prediction(cls, prediction: dspy.Prediction) -> dspy.Prediction:
        """Validate and normalize prediction payload via Pydantic schema contract."""
        @safe
        def _build_typed_history(
            history_raw: dict[str, Any],
        ) -> dict[str, AgentHistory]:
            return {
                agent_name: AgentHistory(
                    turns=[cls._coerce_turn(turn, agent_name) for turn in turns]
                )
                for agent_name, turns in history_raw.items()
            }

        @safe
        def _make_envelope(
            typed_history: dict[str, AgentHistory],
            final_raw: dict[str, Any],
        ) -> MASPredictionEnvelope:
            envelope = MASPredictionEnvelope(
                history=typed_history,
                final_answers=FinalAnswers(values=final_raw),
            )
            cls._validate_envelope(envelope)  # raises ValueError/TypeError on invalid input
            return envelope

        result: Result[dspy.Prediction, Exception] = Result.do(
            dspy.Prediction(**cls._build_normalized_payload(prediction, envelope))
            for history_raw in safe(cls._require_mapping_field)(prediction, 'history')
            for final_raw in safe(cls._require_mapping_field)(prediction, 'final_answers')
            for typed_history in _build_typed_history(history_raw)
            for envelope in _make_envelope(typed_history, final_raw)
        )
        return result.alt(raise_exception).unwrap()

    def __call__(self, **inputs: Any) -> dspy.Prediction:
        """Forward a normalized input mapping to ``program.forward``."""
        # Forward only the fields the MASProgram expects
        prediction = self.program(
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

        if prediction is None or not isinstance(prediction, dspy.Prediction):
            raise RuntimeError(f'MASProgram returned invalid prediction object: {prediction!r}')

        return self._normalize_prediction(prediction)
