"""Typed prediction contracts for MAS evaluation output validation.

The Pydantic models in this file own the entire validation surface for
the payload a ``MASProgram`` emits — both shape coercion (accepting
``dspy.Prediction`` instances *or* plain dicts for individual turns) and
cross-field consistency checks (history non-empty, every agent has
turns, history keys match final-answer keys).  Centralising that here
keeps :class:`bias_mitigation.mas.evaluation.adapters.PredictFnAdapter`
to a thin "extract raw dicts → ``model_validate`` → rebuild
``dspy.Prediction``" pipeline.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


class AgentTurn(BaseModel):
    """One agent response for a single turn in MAS execution history."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: StrictStr
    reasoning: StrictStr
    agent_name: StrictStr | None = None


def _coerce_turn(turn: Any, fallback_agent_name: str) -> dict[str, Any] | Any:
    """Normalize a turn (``dspy.Prediction`` / dict / model) into a dict.

    Default ``agent_name`` to the parent history key when the turn itself
    doesn't carry one.  Already-typed :class:`AgentTurn` instances are
    passed through unchanged so re-validating a model doesn't break it.
    """
    match turn:
        case AgentTurn():
            return turn
        case dict():
            return {
                'answer': turn.get('answer'),
                'reasoning': turn.get('reasoning'),
                'agent_name': turn.get('agent_name', fallback_agent_name) or fallback_agent_name,
            }
        case _:
            return {
                'answer': getattr(turn, 'answer', None),
                'reasoning': getattr(turn, 'reasoning', None),
                'agent_name': getattr(turn, 'agent_name', None) or fallback_agent_name,
            }


class AgentHistory(BaseModel):
    """Ordered turn history for one agent."""

    turns: list[AgentTurn] = Field(default_factory=list)


class FinalAnswers(BaseModel):
    """Final answer mapping keyed by agent name."""

    values: dict[str, StrictStr] = Field(default_factory=dict)


class MASPredictionEnvelope(BaseModel):
    """Validated shape expected by evaluator scorers.

    The ``mode='before'`` validator accepts the raw ``history`` dict the
    MAS program emits (values may be plain lists of turns, lists with
    mixed dict / ``dspy.Prediction`` shapes, or already-typed
    :class:`AgentHistory` instances) and normalises each turn through
    :func:`_coerce_turn`, defaulting any missing ``agent_name`` to the
    outer history key.  The ``mode='after'`` validator enforces the
    cross-field invariants downstream scorers rely on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history: dict[str, AgentHistory] = Field(default_factory=dict)
    final_answers: FinalAnswers

    @model_validator(mode='before')
    @classmethod
    def _normalize_history_turns(cls, data: Any) -> Any:
        """Coerce raw history into a shape :class:`AgentHistory` accepts."""
        if not isinstance(data, dict) or not isinstance(data.get('history'), dict):
            return data
        normalized: dict[str, Any] = {}
        for agent_name, agent_history in data['history'].items():
            match agent_history:
                case AgentHistory():
                    normalized[agent_name] = agent_history
                case list():
                    normalized[agent_name] = {
                        'turns': [_coerce_turn(t, agent_name) for t in agent_history],
                    }
                case {'turns': turns} if isinstance(turns, list):
                    normalized[agent_name] = {
                        'turns': [_coerce_turn(t, agent_name) for t in turns],
                    }
                case _:
                    normalized[agent_name] = agent_history
        return {**data, 'history': normalized}

    @model_validator(mode='after')
    def _check_consistency(self) -> 'MASPredictionEnvelope':
        """Enforce non-empty history and matching agent keys."""
        if not self.history:
            raise ValueError('Prediction history is empty; cannot score MAS metrics.')
        for agent_name, agent_history in self.history.items():
            if not agent_history.turns:
                raise ValueError(f'Prediction history for {agent_name} has no turns.')
        history_keys = set(self.history.keys())
        final_keys = set(self.final_answers.values.keys())
        if history_keys != final_keys:
            raise ValueError(
                'Mismatch between history agent keys and final_answers keys: '
                f'history={sorted(history_keys)}, final_answers={sorted(final_keys)}',
            )
        return self

    def as_plain_history(self) -> dict[str, list[AgentTurn]]:
        """Return history as plain mapping used by adapter reconstruction."""
        return {
            agent_name: agent_history.turns for agent_name, agent_history in self.history.items()
        }
