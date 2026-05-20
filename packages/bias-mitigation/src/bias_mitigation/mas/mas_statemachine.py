"""State-machine orchestration for preregistered MAS evaluation runs.

This module defines the deterministic lifecycle used in experiments:

1. Genesis phase: each agent produces an initial answer.
2. Interaction phases: agents iteratively update answers using peer context.
3. Completion: per-agent prediction history is returned for downstream metrics.
"""

import asyncio
import re
from itertools import starmap

import dspy
from loguru import logger
from returns.result import Failure, Result, Success
from statemachine import State, StateChart

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.protocols import ProtocolStrategy

from .agent import Agent, AgentExecutionError

PhaseResult = Result[dspy.Prediction, AgentExecutionError]


class MASStateMachine(StateChart[Agent]):
    """Declarative lifecycle: genesis → interaction x N → completed.

    The whole lifecycle collapses into a single :attr:`advance` event:
    python-statemachine evaluates the :meth:`rounds_exhausted` guard and
    picks the right edge (continue the interaction loop or finish), so
    the async ``on_enter_*`` handlers never have to know which kind of
    transition they are taking next.
    """

    catch_errors_as_events = False  # fail-fast

    genesis = State(initial=True, value='genesis')
    interaction = State(value='interaction')
    completed = State(final=True, value='completed')

    advance = (
        genesis.to(interaction)
        | interaction.to(interaction, unless='rounds_exhausted')
        | interaction.to(completed, cond='rounds_exhausted')
    )

    def rounds_exhausted(self) -> bool:
        return self.current_round > self.config.rounds

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

    def _format_peer_answers(
        self,
        agent: Agent,
        prev_answers: dict[str, dspy.Prediction],
    ) -> str:
        return '\n'.join(
            f'{p_name}: {self._prediction_answer(prev_answers[p_name])} — '
            f'{self._prediction_reasoning(prev_answers[p_name])}'
            for p_name in self._history
            if p_name != agent.name
        )

    def _logging_context(self, phase: str) -> dict[str, str | int]:
        return {
            'artifact_root': self._artifact_root(),
            'sample_id': self.sample_id,
            'round_index': self.current_round,
            'phase': phase,
        }

    @staticmethod
    def _collect_prediction_result(
        agent: Agent,
        result: PhaseResult,
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
                logger.error(
                    f'Agent {agent.name} failed during {phase} after retries: {failure.reason}'
                )
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
        self.current_round = 0

        super().__init__()
        # python-statemachine detects the async on_enter_* handlers and
        # defers the lifecycle — caller must drive it with
        # ``await sm.activate_initial_state()``.

    async def _genesis_call(self, agent_idx: int, agent: Agent) -> PhaseResult:
        return await agent.aforward(
            question=self.question,
            context=self.context,
            options=self.options,
            system_prompt=self.protocol.get_system_prompt(self.groups[agent_idx]),
            logging_context=self._logging_context('genesis'),
        )

    async def _interaction_call(
        self,
        agent_idx: int,
        agent: Agent,
        prev_answers: dict[str, dspy.Prediction],
    ) -> PhaseResult:
        return await agent.aforward(
            question=self.question,
            context=self.context,
            options=self.options,
            system_prompt=self.protocol.get_system_prompt(self.groups[agent_idx]),
            peer_answers=self._format_peer_answers(agent, prev_answers),
            update_instruction=self.protocol.get_update_instruction(),
            logging_context=self._logging_context('interaction'),
        )

    async def on_enter_genesis(self) -> None:
        """Run all agents' genesis turn concurrently, then advance."""
        results = await asyncio.gather(
            *starmap(self._genesis_call, enumerate(self.agents)),
        )
        for agent, pred in zip(self.agents, self._validate_phase('genesis', results), strict=True):
            self._history[agent.name].append(pred)
        await self.advance()

    async def on_enter_interaction(self) -> None:
        """Run all agents' update turn concurrently."""
        self.current_round += 1
        if self.rounds_exhausted():
            await self.advance()
            return

        prev_answers = {name: preds[-1] for name, preds in self._history.items()}
        results = await asyncio.gather(
            *(self._interaction_call(i, a, prev_answers) for i, a in enumerate(self.agents)),
        )
        for agent, pred in zip(
            self.agents, self._validate_phase('interaction', results), strict=True
        ):
            self._history[agent.name].append(pred)
        await self.advance()

    async def on_enter_completed(self) -> None:
        """Cleanup hook fired when the lifecycle terminates.

        Marks each agent's own state machine as completed; previously
        this was inlined into ``on_enter_interaction``'s exhausted branch.
        """
        for agent in self.agents:
            if hasattr(agent, 'lifecycle'):
                agent.lifecycle.mark_completed()

    def _validate_phase(
        self,
        phase: str,
        results: list[PhaseResult],
    ) -> list[dspy.Prediction]:
        """Split phase results into predictions and failures; raise on any failure."""
        predictions: list[dspy.Prediction] = []
        failures: list[AgentExecutionError] = []
        for agent, result in zip(self.agents, results, strict=True):
            self._collect_prediction_result(agent, result, phase, predictions, failures)
        self._raise_phase_failures(phase, failures)
        return predictions

    @property
    def history(self) -> dict[str, list[dspy.Prediction]]:
        """Return the per-agent prediction history accumulated so far."""
        return self._history
