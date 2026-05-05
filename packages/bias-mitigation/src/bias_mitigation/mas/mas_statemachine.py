"""State-machine orchestration for preregistered MAS evaluation runs.

This module defines the deterministic lifecycle used in experiments:

1. Genesis phase: each agent produces an initial answer.
2. Interaction phases: agents iteratively update answers using peer context.
3. Completion: per-agent prediction history is returned for downstream metrics.
"""

import re

import dspy
from loguru import logger
from mlflow.entities import SpanType
from returns.result import Failure, Success
from statemachine import State, StateChart

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.protocols import ProtocolStrategy

from .agent import Agent, AgentExecutionError


class MASStateMachine(StateChart[Agent]):
    """Declarative lifecycle: genesis → interaction × N → completed."""

    catch_errors_as_events = False  # enterprise fail-fast

    # States
    genesis = State(initial=True)
    interaction = State()
    completed = State(final=True)

    # Declarative transitions (library-managed event queue)
    to_interaction = genesis.to(interaction)
    continue_interaction = interaction.to(interaction)
    finish = interaction.to(completed)

    @staticmethod
    def _prediction_reasoning(prediction: dspy.Prediction) -> str:
        reasoning = getattr(prediction, 'reasoning', None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return 'No reasoning provided.'

    @staticmethod
    def _prediction_answer(prediction: dspy.Prediction) -> str:
        answer = getattr(prediction, 'answer', None)
        if isinstance(answer, str) and answer.strip():
            return answer
        return 'Unknown'

    @staticmethod
    def _sanitize_path_token(token: str) -> str:
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', token).strip('_') or 'unknown'

    def _artifact_root(self) -> str:
        return f'agent_logs/{self.run_id}/{self._sanitize_path_token(self.sample_id)}'

    @staticmethod
    def _raise_phase_failures(phase: str, failures: list[AgentExecutionError]) -> None:
        if not failures:
            return
        summary = '; '.join(
            f'{failure.agent_name} ({failure.phase}): {failure.reason}' for failure in failures
        )
        raise RuntimeError(f'MAS {phase} failed after retries: {summary}')

    def _interaction_kwargs(
        self,
        agent: Agent,
        agent_idx: int,
        prev_answers: dict[str, dspy.Prediction],
    ) -> dict[str, str | list[str] | dict[str, str | int]]:
        return {
            'question': self.question,
            'context': self.context,
            'options': self.options,
            'system_prompt': self.protocol.get_system_prompt(self.groups[agent_idx]),
            'peer_answers': '\n'.join(
                f'{p_name}: {self._prediction_answer(prev_answers[p_name])} — '
                f'{self._prediction_reasoning(prev_answers[p_name])}'
                for p_name in self._history
                if p_name != agent.name
            ),
            'update_instruction': self.protocol.get_update_instruction(),
            'logging_context': {
                'artifact_root': self._artifact_root(),
                'sample_id': self.sample_id,
                'round_index': self.current_round,
                'phase': 'interaction',
            },
        }

    @staticmethod
    def _collect_prediction_result(
        agent: Agent,
        result: Success[dspy.Prediction] | Failure[AgentExecutionError],
        phase: str,
        predictions: list[dspy.Prediction],
        failures: list[AgentExecutionError],
    ) -> None:
        match result:
            case Success(prediction):
                if not hasattr(prediction, 'answer') or not isinstance(prediction.answer, str):
                    failures.append(
                        AgentExecutionError(
                            agent_name=agent.name,
                            phase=phase,
                            reason=f'Invalid prediction payload: {prediction}',
                        )
                    )
                else:
                    predictions.append(prediction)
            case Failure(failure):
                logger.error(f'Agent {agent.name} failed during {phase} after retries: {failure.reason}')
                failures.append(failure)

    def __init__(
        self,
        agents: list[Agent],
        options: list[str],
        groups: list[str],
        context: str,
        question: str,
        protocol: ProtocolStrategy,
        config: MASConfig,
        sample_id: str,
        run_id: str,
    ):
        self.agents = agents
        self._history: dict[str, list[dspy.Prediction]] = {a.name: [] for a in agents}
        self.options = options
        self.groups = groups
        self.context = context
        self.question = question
        self.protocol = protocol
        self.config = config
        self.sample_id = sample_id
        self.run_id = run_id
        self.genesis_executor = None
        self.update_executor = None
        self.current_round = 0
        self._genesis_pairs = [
            (
                agent,
                {
                    'question': self.question,
                    'context': self.context,
                    'options': self.options,
                    'system_prompt': self.protocol.get_system_prompt(self.groups[i]),
                    'logging_context': {
                        'artifact_root': self._artifact_root(),
                        'sample_id': self.sample_id,
                        'round_index': 0,
                        'phase': 'genesis',
                    },
                },
            )
            for i, agent in enumerate(self.agents)
        ]

        super().__init__()  # triggers on_enter_genesis immediately → full lifecycle runs declaratively

    def on_enter_genesis(self, target, event):
        genesis_results: list[dspy.Prediction] = []
        failures: list[AgentExecutionError] = []
        for agent, kwargs in self._genesis_pairs:
            res = agent(**kwargs)
            self._collect_prediction_result(
                agent=agent,
                result=res,
                phase='genesis',
                predictions=genesis_results,
                failures=failures,
            )

        self._raise_phase_failures('genesis', failures)

        for agent, pred in zip(self.agents, genesis_results, strict=False):
            self._history[agent.name].append(pred)

        self.to_interaction()  # declarative chain to interaction phase

    def on_enter_interaction(self, target, event):
        self.current_round += 1

        if self.current_round > self.config.rounds:
            for agent in self.agents:
                if hasattr(agent, 'lifecycle'):
                    agent.lifecycle.mark_completed()
            self.finish()
            return

        prev_answers = {name: preds[-1] for name, preds in self._history.items()}

        update_pairs = [
            (agent, self._interaction_kwargs(agent, agent_idx, prev_answers))
            for agent_idx, agent in enumerate(self.agents)
        ]

        new_preds: list[dspy.Prediction] = []
        failures: list[AgentExecutionError] = []
        for agent, kwargs in update_pairs:
            res = agent(**kwargs)
            self._collect_prediction_result(agent, res, 'interaction', new_preds, failures)

        self._raise_phase_failures('interaction', failures)

        for agent, pred in zip(self.agents, new_preds, strict=False):
            self._history[agent.name].append(pred)

        self.continue_interaction()  # library-managed recursive event queue

    def run(self) -> dict[str, list[dspy.Prediction]]:
        return self._history
