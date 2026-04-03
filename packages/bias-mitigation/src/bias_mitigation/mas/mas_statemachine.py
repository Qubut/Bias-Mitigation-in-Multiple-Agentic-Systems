"""State-machine orchestration for preregistered MAS evaluation runs.

This module defines the deterministic lifecycle used in experiments:

1. Genesis phase: each agent produces an initial answer.
2. Interaction phases: agents iteratively update answers using peer context.
3. Completion: per-agent prediction history is returned for downstream metrics.
"""

import dspy
import mlflow
from mlflow.entities import SpanType
from statemachine import State, StateChart

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.protocols import ProtocolStrategy

from .agent import Agent


class MASStateMachine(StateChart[Agent]):
    """Declarative lifecycle engine for a single MAS evaluation run.

    States:
        - ``genesis``: generate turn-0 prediction per agent.
        - ``interaction``: update predictions round-by-round from peer context.
        - ``completed``: terminal state exposing final history.

    Transitions:
        - ``genesis -> interaction`` via ``to_interaction``.
        - ``interaction -> interaction`` via ``continue_interaction``.
        - ``interaction -> completed`` via ``finish``.

    The output history is consumed downstream to compute propagation,
    emergence, amplification, and robustness metrics.
    """

    catch_errors_as_events = False  # enterprise fail-fast

    # States
    genesis = State(initial=True)
    interaction = State()
    completed = State(final=True)

    # Declarative transitions (library-managed event queue)
    to_interaction = genesis.to(interaction)
    continue_interaction = interaction.to(interaction)
    finish = interaction.to(completed)

    def __init__(
        self,
        agents: list[Agent],
        options: list[str],
        groups: list[str],
        context: str,
        question: str,
        protocol: ProtocolStrategy,
        config: MASConfig,
        genesis_executor: dspy.Parallel,
        update_executor: dspy.Parallel,
    ):
        """Initialize machine inputs and trigger lifecycle execution.

        Args:
            agents: Ordered list of agent instances participating in the run.
            options: Multiple-choice answer options for the current sample.
            groups: Group labels aligned to agents for prompt conditioning.
            context: Input context passage for the sample.
            question: Prompt question for the sample.
            protocol: Protocol strategy defining prompts and update instructions.
            config: Runtime configuration including number of interaction rounds.
            genesis_executor: Executor for genesis inference calls.
            update_executor: Executor for interaction-round inference calls.

        Side Effects:
            Calling ``super().__init__()`` enters the initial state and starts the
            genesis -> interaction lifecycle automatically.
        """
        self.agents = agents
        self._history: dict[str, list[dspy.Prediction]] = {a.name: [] for a in agents}
        self.options = options
        self.groups = groups
        self.context = context
        self.question = question
        self.protocol = protocol
        self.config = config
        self.genesis_executor = genesis_executor
        self.update_executor = update_executor
        self.current_round = 0
        self._genesis_pairs = [
            (
                agent,
                {
                    'question': self.question,
                    'context': self.context,
                    'options': self.options,
                    'system_prompt': self.protocol.get_system_prompt(self.groups[i]),
                },
            )
            for i, agent in enumerate(self.agents)
        ]

        super().__init__()  # triggers on_enter_genesis immediately → full lifecycle runs declaratively

    def on_enter_genesis(self, target, event):
        """Run the genesis phase once and seed per-agent history.

        Each agent receives question/context/options plus its group-conditioned
        system prompt. The resulting first prediction defines turn-0 outputs used
        as the baseline for subsequent interaction metrics.

        Side Effects:
            - Appends one prediction per agent to ``self._history``.
            - Emits MLflow span metadata for the genesis phase.
            - Triggers ``to_interaction`` transition.
        """
        with mlflow.start_span(name='MAS_Genesis', span_type=SpanType.AGENT) as span:
            span.set_attribute('phase', 'genesis')
            span.set_attribute('memory.reset_on_genesis', self.config.reset_memory_on_genesis)

            if self.config.reset_memory_on_genesis:
                memory_clear_attempts = 0
                memory_clear_successes = 0
                for agent in self.agents:
                    if getattr(agent, 'memory_client', None):
                        memory_clear_attempts += 1
                        agent.memory_client.clear_user_memory(user_id=agent.user_id)
                        memory_clear_successes += 1
                span.set_attribute('memory.clear.attempts', memory_clear_attempts)
                span.set_attribute('memory.clear.successes', memory_clear_successes)
                mlflow.log_metric('memory.clear.attempts', float(memory_clear_attempts))
                mlflow.log_metric('memory.clear.successes', float(memory_clear_successes))

            genesis_results = self.genesis_executor(self._genesis_pairs)

            for agent, pred in zip(self.agents, genesis_results, strict=False):
                self._history[agent.name].append(pred)

        self.to_interaction()  # declarative chain to interaction phase

    def on_enter_interaction(self, target, event):
        """Run one interaction round and append updated predictions.

        For each round, each agent receives peer answers from the previous round
        and an update instruction from the configured protocol strategy.
        The state loops until ``current_round > config.rounds``, then transitions
        to ``completed``.

        Side Effects:
            - Increments ``self.current_round``.
            - Appends one updated prediction per agent for each executed round.
            - Emits MLflow span metadata for round-level observability.
        """
        self.current_round += 1

        if self.current_round > self.config.rounds:
            for agent in self.agents:
                if hasattr(agent, 'lifecycle'):
                    agent.lifecycle.mark_completed()
            self.finish()
            return

        with mlflow.start_span(
            name=f'MAS_Round_{self.current_round}', span_type=SpanType.AGENT
        ) as span:
            span.set_attribute('round', self.current_round)

            prev_answers = {name: preds[-1] for name, preds in self._history.items()}

            update_pairs = [
                (
                    agent,
                    {
                        'question': self.question,
                        'context': self.context,
                        'options': self.options,
                        'system_prompt': self.protocol.get_system_prompt(self.groups[agent_idx]),
                        'peer_answers': '\n'.join(
                            f'{p_name}: {prev_answers[p_name].answer} — {prev_answers[p_name].reasoning}'
                            for p_name in self._history
                            if p_name != agent.name
                        ),
                        'update_instruction': self.protocol.get_update_instruction(),
                    },
                )
                for agent_idx, agent in enumerate(self.agents)
            ]

            new_preds = self.update_executor(update_pairs)

            for agent, pred in zip(self.agents, new_preds, strict=False):
                self._history[agent.name].append(pred)

        self.continue_interaction()  # library-managed recursive event queue

    def run(self) -> dict[str, list[dspy.Prediction]]:
        """Return per-agent prediction history for the full run.

        Returns:
            Mapping from agent name to ordered predictions, where index ``0`` is
            the genesis output and indices ``1..N`` are interaction-round outputs.
        """
        return self._history
