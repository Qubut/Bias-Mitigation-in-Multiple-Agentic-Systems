import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dspy
import mlflow
from loguru import logger
from mlflow.entities import SpanType
from returns.result import Failure, Result, Success
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from bias_mitigation.mas.agent_statemachine import AgentState, AgentStateMachine
from bias_mitigation.mas.signatures import InitialAnswer, UpdateAnswer, UpdateAnswerWithMemory
from bias_mitigation.memory import Mem0Tools


@dataclass(frozen=True)
class AgentExecutionError:
    """Typed agent execution failure envelope for declarative error propagation."""

    agent_name: str
    phase: str
    reason: str


class Agent(dspy.Module):
    """Per-agent reasoning module with optional memory-backed updates."""

    def __init__(
        self,
        name: str,
        group: str,
        lm: dspy.LM,
        memory_tools: Mem0Tools | None = None,
        run_id: str | None = None,
    ):
        """Initialize one agent runtime instance.

        Args:
            name: Agent identifier used in history and tracing.
            group: Social-group label used for prompt conditioning.
            lm: DSPy language model instance.
            memory_tools: Optional memory tooling adapter used in memory interventions.
            run_id: Optional run-scoped identifier for session isolation.
        """
        super().__init__()
        self.name = name
        self.group = group
        self.lm = lm
        self.memory_tools = memory_tools
        self.run_id = run_id or str(uuid.uuid4())
        self.user_id = f'{self.name}_{self.run_id}'
        self.lifecycle = AgentStateMachine()

        with mlflow.start_span(name=f'Agent_Init_{name}', span_type=SpanType.AGENT) as span:
            span.set_attribute('agent.name', name)
            span.set_attribute('agent.group', group)

        with dspy.context(lm=self.lm):
            self.initial = dspy.Predict(InitialAnswer)
            self.update = dspy.Predict(UpdateAnswerWithMemory if memory_tools else UpdateAnswer)

    @staticmethod
    def _normalize_reasoning(raw_reasoning: Any) -> str:
        if isinstance(raw_reasoning, str) and raw_reasoning.strip():
            return raw_reasoning
        return 'No reasoning provided.'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _predict_initial(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
    ) -> Any:
        return self.initial(
            question=question,
            context=context,
            options=options,
            system_prompt=system_prompt,
            group=self.group,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _predict_update(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str,
        update_instruction: str,
        recalled_memory: str,
    ) -> Any:
        return self.update(
            question=question,
            context=context,
            options=options,
            system_prompt=system_prompt,
            peer_answers=peer_answers,
            past_interaction_memory=recalled_memory,
            protocol_instruction=update_instruction,
            group=self.group,
        )

    def _run_with_retry(
        self,
        predictor_call: Callable[[], Any],
        phase: AgentState,
    ) -> Result[Any, AgentExecutionError]:
        try:
            return Success(predictor_call())
        except Exception as error:
            return Failure(
                AgentExecutionError(
                    agent_name=self.name,
                    phase=phase.value,
                    reason=str(error),
                )
            )

    def _recall_memory(self, question: str) -> str:
        if not self.memory_tools:
            return 'No previous statements found.'

        results = self.memory_tools.search_memories(query=question, user_id=self.user_id, limit=3)
        passages: list[str] = []
        match results:
            case Success(memories):
                passages = memories.get('passages', [])
            case Failure(error):
                logger.error(f'[Agent]: Failed to search memories: {error}')
                passages = []
            case _:
                passages = []
        return '\n'.join(passages) or 'No previous statements found.'

    def _store_memory(self, answer: str, reasoning: str) -> None:
        if not self.memory_tools:
            return

        store_result = self.memory_tools.store_memory(
            content=f'My previous answer: {answer}. Reasoning: {reasoning}',
            user_id=self.user_id,
            metadata={'agent': self.name},
        )
        if isinstance(store_result, Failure):
            logger.warning(f'[Agent]: Failed to store memory: {store_result.failure()}')

    @mlflow.trace(name='Agent_Forward', span_type=SpanType.AGENT)
    def forward(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str | None = None,
        update_instruction: str | None = None,
    ) -> Result[dspy.Prediction, AgentExecutionError]:
        """Run one agent step for genesis or interaction phases.

        If ``peer_answers`` is absent, the agent produces a genesis response.
        Otherwise it performs an interaction update, optionally using recalled
        memory when a memory client is configured.
        """
        phase = self.lifecycle.transition_for_step(has_peer_answers=peer_answers is not None)

        input_payload = {
            'question': question,
            'context': context or '',
            'options': options,
            'system_prompt': system_prompt,
            'peer_answers': peer_answers,
            'update_instruction': update_instruction,
            'agent_name': self.name,
            'agent_group': self.group,
            'agent_phase': phase,
        }
        mlflow.log_dict(input_payload, f'agent_{self.name}_inputs.json')

        with dspy.context(lm=self.lm):
            match phase:
                case AgentState.GENESIS:
                    prediction_result = self._run_with_retry(
                        lambda: self._predict_initial(
                            question=question,
                            context=context,
                            options=options,
                            system_prompt=system_prompt,
                        ),
                        phase=phase,
                    )
                case _:
                    recalled = self._recall_memory(question)
                    prediction_result = self._run_with_retry(
                        lambda: self._predict_update(
                            question=question,
                            context=context,
                            options=options,
                            system_prompt=system_prompt,
                            peer_answers=peer_answers or '',
                            update_instruction=update_instruction or '',
                            recalled_memory=recalled,
                        ),
                        phase=phase,
                    )

        if isinstance(prediction_result, Failure):
            error = prediction_result.failure()
            logger.error(
                f'[Agent]: Prediction failed after retries for {error.agent_name} '
                f'(phase={error.phase}): {error.reason}'
            )
            return prediction_result

        pred = prediction_result.unwrap()
        answer = getattr(pred, 'answer', None)
        if not isinstance(answer, str) or not answer.strip():
            return Failure(
                AgentExecutionError(
                    agent_name=self.name,
                    phase=phase.value,
                    reason=f'Invalid answer payload returned: {pred}',
                )
            )

        reasoning = self._normalize_reasoning(getattr(pred, 'reasoning', None))
        self._store_memory(answer, reasoning)

        output_payload = {
            'answer': answer,
            'reasoning': reasoning,
            'agent_name': self.name,
        }
        mlflow.log_dict(output_payload, f'agent_{self.name}_output.json')

        if hasattr(self.lm, 'last_token_usage'):
            mlflow.log_metric('tokens_used', getattr(self.lm, 'last_token_usage', 0))

        return Success(dspy.Prediction(answer=answer, reasoning=reasoning, agent_name=self.name))
