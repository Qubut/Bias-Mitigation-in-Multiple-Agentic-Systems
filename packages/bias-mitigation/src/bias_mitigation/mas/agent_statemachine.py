"""Lifecycle state machine for one MAS agent execution context."""

from enum import StrEnum

from statemachine import State, StateChart


class AgentState(StrEnum):
    """States in a single-agent lifecycle for one sample run."""

    GENESIS = 'genesis'
    INTERACTION = 'interaction'
    COMPLETED = 'completed'


class AgentStateMachine(StateChart[None]):
    """Small explicit state machine controlling per-agent phase transitions."""

    genesis = State(initial=True)
    interaction = State()
    completed = State(final=True)

    start_interaction = genesis.to(interaction)
    continue_interaction = interaction.to(interaction)
    finish = genesis.to(completed) | interaction.to(completed)

    def __init__(self) -> None:
        super().__init__()

    def _active_state(self) -> AgentState:
        """Return the currently active state for this agent lifecycle."""
        return AgentState(self.configuration[0].id)

    def transition_for_step(self, *, has_peer_answers: bool) -> AgentState:
        """Advance state based on whether this step is genesis or interaction."""
        current_id = self._active_state()
        if current_id == AgentState.COMPLETED:
            raise RuntimeError('Cannot transition a completed agent state machine.')

        if has_peer_answers:
            if current_id == AgentState.GENESIS:
                self.start_interaction()
            else:
                self.continue_interaction()
        elif current_id != AgentState.GENESIS:
            raise RuntimeError('Genesis step cannot run after interaction has started.')

        return self._active_state()

    def mark_completed(self) -> None:
        """Mark the lifecycle terminal state."""
        if self._active_state() != AgentState.COMPLETED:
            self.finish()
