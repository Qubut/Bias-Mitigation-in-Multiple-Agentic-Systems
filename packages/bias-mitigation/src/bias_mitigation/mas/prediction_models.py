"""Typed prediction contracts for MAS evaluation output validation."""

from pydantic import BaseModel, Field, StrictStr


class AgentTurn(BaseModel):
    """One agent response for a single turn in MAS execution history."""

    answer: StrictStr
    reasoning: StrictStr
    agent_name: StrictStr | None = None


class AgentHistory(BaseModel):
    """Ordered turn history for one agent."""

    turns: list[AgentTurn] = Field(default_factory=list)


class FinalAnswers(BaseModel):
    """Final answer mapping keyed by agent name."""

    values: dict[str, StrictStr] = Field(default_factory=dict)


class MASPredictionEnvelope(BaseModel):
    """Validated shape expected by evaluator scorers."""

    history: dict[str, AgentHistory] = Field(default_factory=dict)
    final_answers: FinalAnswers

    def as_plain_history(self) -> dict[str, list[AgentTurn]]:
        """Return history as plain mapping used by adapter reconstruction."""
        return {agent_name: agent_history.turns for agent_name, agent_history in self.history.items()}
