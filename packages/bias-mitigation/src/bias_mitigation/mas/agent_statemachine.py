"""Per-agent lifecycle + LM dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import BoundedSemaphore, Lock

import dspy
from pydantic import BaseModel, ConfigDict
from returns.result import Failure, Result, Success
from statemachine import State, StateChart
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class AgentState(StrEnum):
    GENESIS = 'genesis'
    INTERACTION = 'interaction'
    COMPLETED = 'completed'


@dataclass(frozen=True)
class AgentExecutionError:
    agent_name: str
    phase: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetryConfig:
    attempts: int
    backoff_min_seconds: float
    backoff_max_seconds: float
    max_inflight_per_endpoint: int


# threading.BoundedSemaphore, not asyncio — the cap must hold across the
# worker loops dspy.syncify spawns; asyncio.Semaphore is loop-local.
_LLM_BULKHEADS: dict[str, BoundedSemaphore] = {}
_LLM_BULKHEAD_LOCK = Lock()


def _llm_bulkhead(endpoint_key: str, limit: int) -> BoundedSemaphore:
    with _LLM_BULKHEAD_LOCK:
        existing = _LLM_BULKHEADS.get(endpoint_key)
        if existing is not None:
            return existing
        semaphore = BoundedSemaphore(limit)
        _LLM_BULKHEADS[endpoint_key] = semaphore
        return semaphore


PredictorCall = Callable[[], dspy.Prediction]


class _InitialInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    context: str
    options: list[str]
    system_prompt: str
    group: str


class _UpdateInputs(BaseModel):

    model_config = ConfigDict(frozen=True)

    question: str
    context: str
    options: list[str]
    system_prompt: str
    group: str
    peer_answers: str
    protocol_instruction: str
    past_interaction_memory: str | None = None


def _predictor_input_fields(predictor: dspy.Predict) -> set[str]:
    signature = getattr(predictor, 'signature', None)
    fields = getattr(signature, 'input_fields', None) or {}
    return set(fields.keys())


class AgentStateMachine(StateChart[None]):
    """One agent, one sample."""

    allow_event_without_transition = False

    genesis = State(initial=True, value=AgentState.GENESIS.value)
    interaction = State(value=AgentState.INTERACTION.value)
    completed = State(final=True, value=AgentState.COMPLETED.value)

    # to(self) instead of to.itself() — the latter trips Pylance (no stubs).
    step = (
        genesis.to(interaction, cond='has_peer_answers')
        | interaction.to(interaction, cond='has_peer_answers')
        | genesis.to(genesis, unless='has_peer_answers')
    )
    finish = genesis.to(completed) | interaction.to(completed)

    def __init__(self) -> None:
        super().__init__()
        self._has_peer_answers = False

    def has_peer_answers(self) -> bool:
        return self._has_peer_answers

    def current_phase(self) -> AgentState:
        return AgentState(self.configuration[0].id)

    def transition_for_step(self, *, has_peer_answers: bool) -> AgentState:
        """Step once. Returns the resulting phase."""
        self._has_peer_answers = has_peer_answers
        self.step()
        return self.current_phase()

    def mark_completed(self) -> None:
        if self.current_phase() is not AgentState.COMPLETED:
            self.finish()

    @staticmethod
    def predictor_call(
        phase: AgentState,
        *,
        initial: dspy.Predict,
        update: dspy.Predict,
        group: str,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str | None,
        update_instruction: str | None,
        recalled_memory: str,
    ) -> PredictorCall:
        if phase is AgentState.GENESIS:
            initial_inputs = _InitialInputs(
                question=question,
                context=context,
                options=options,
                system_prompt=system_prompt,
                group=group,
            )
            return lambda: initial(**initial_inputs.model_dump())

        update_inputs = _UpdateInputs(
            question=question,
            context=context,
            options=options,
            system_prompt=system_prompt,
            group=group,
            peer_answers=peer_answers or '',
            protocol_instruction=update_instruction or '',
            past_interaction_memory=(
                recalled_memory
                if 'past_interaction_memory' in _predictor_input_fields(update)
                else None
            ),
        )
        payload = update_inputs.model_dump(exclude_none=True)
        return lambda: update(**payload)

    @staticmethod
    async def run_predictor(
        predictor_call: PredictorCall,
        *,
        phase: AgentState,
        agent_name: str,
        endpoint_key: str,
        retry_config: RetryConfig,
    ) -> Result[dspy.Prediction, AgentExecutionError]:
        semaphore = _llm_bulkhead(endpoint_key, retry_config.max_inflight_per_endpoint)

        @retry(
            stop=stop_after_attempt(retry_config.attempts),
            wait=wait_exponential(
                multiplier=1,
                min=retry_config.backoff_min_seconds,
                max=retry_config.backoff_max_seconds,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def run_once() -> dspy.Prediction:
            await asyncio.to_thread(semaphore.acquire)
            try:
                return await asyncio.to_thread(predictor_call)
            finally:
                semaphore.release()

        try:
            return Success(await run_once())
        except Exception as error:
            return Failure(
                AgentExecutionError(
                    agent_name=agent_name,
                    phase=phase.value,
                    reason=str(error),
                )
            )
