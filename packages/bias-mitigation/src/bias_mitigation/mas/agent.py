import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
from typing import Any

import dspy
import mlflow
from loguru import logger
from returns.result import Failure, Result, Success
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from bias_mitigation.mas.agent_statemachine import AgentState, AgentStateMachine
from bias_mitigation.mas.signatures import InitialAnswer, UpdateAnswer, UpdateAnswerWithMemory
from bias_mitigation.memory import Mem0Tools
from bias_mitigation.memory.orchestration.service import MemoryOrchestrator


@dataclass(frozen=True)
class AgentExecutionError:
    """Typed agent execution failure envelope for declarative error propagation."""

    agent_name: str
    phase: str
    reason: str


class Agent(dspy.Module):
    """Per-agent reasoning module with optional memory-backed updates."""

    _LLM_BULKHEADS: dict[str, BoundedSemaphore] = {}
    _LLM_BULKHEAD_LOCK = Lock()

    def __init__(
        self,
        name: str,
        group: str,
        lm: dspy.LM,
        memory_tools: Mem0Tools | None = None,
        memory_orchestrator: MemoryOrchestrator | None = None,
        run_id: str | None = None,
        enable_runtime_artifacts: bool = False,
        llm_max_inflight_per_endpoint: int = 3,
        llm_retry_attempts: int = 3,
        llm_retry_backoff_min_seconds: float = 1.0,
        llm_retry_backoff_max_seconds: float = 4.0,
    ):
        super().__init__()
        self.name = name
        self.group = group
        self.lm = lm
        self.memory_tools = memory_tools
        self.memory_orchestrator = memory_orchestrator
        self.run_id = run_id or str(uuid.uuid4())
        active_run = mlflow.active_run()
        experiment_run_id = (
            active_run.info.run_id if active_run is not None and active_run.info is not None else self.run_id
        )
        self.memory_scope = f'experiment:{experiment_run_id}'
        self.user_id = f'{self.name}_{self.run_id}'
        self.lifecycle = AgentStateMachine()
        self.enable_runtime_artifacts = enable_runtime_artifacts
        self.llm_max_inflight_per_endpoint = max(1, llm_max_inflight_per_endpoint)
        self.llm_retry_attempts = max(1, llm_retry_attempts)
        self.llm_retry_backoff_min_seconds = max(0.0, llm_retry_backoff_min_seconds)
        self.llm_retry_backoff_max_seconds = max(
            self.llm_retry_backoff_min_seconds,
            llm_retry_backoff_max_seconds,
        )

        memory_enabled = memory_tools is not None or memory_orchestrator is not None
        with dspy.context(lm=self.lm):
            self.initial = dspy.Predict(InitialAnswer)
            self.update = dspy.Predict(UpdateAnswerWithMemory if memory_enabled else UpdateAnswer)

    @staticmethod
    def _normalize_reasoning(raw_reasoning: Any) -> str:
        if isinstance(raw_reasoning, str) and raw_reasoning.strip():
            return raw_reasoning
        return 'No reasoning provided.'

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
        retryer = Retrying(
            stop=stop_after_attempt(self.llm_retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self.llm_retry_backoff_min_seconds,
                max=self.llm_retry_backoff_max_seconds,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

        def run_once() -> Any:
            with self._llm_slot():
                return predictor_call()

        try:
            return Success(retryer(run_once))
        except Exception as error:
            return Failure(
                AgentExecutionError(
                    agent_name=self.name,
                    phase=phase.value,
                    reason=str(error),
                )
            )

    def _llm_endpoint_key(self) -> str:
        model_name = str(getattr(self.lm, 'model', 'unknown-model'))
        api_base = str(getattr(self.lm, 'api_base', 'unknown-base'))
        return f'{api_base}|{model_name}'

    @classmethod
    def _llm_bulkhead(cls, endpoint_key: str, limit: int) -> BoundedSemaphore:
        with cls._LLM_BULKHEAD_LOCK:
            existing = cls._LLM_BULKHEADS.get(endpoint_key)
            if existing is not None:
                return existing

            semaphore = BoundedSemaphore(limit)
            cls._LLM_BULKHEADS[endpoint_key] = semaphore
            return semaphore

    @contextmanager
    def _llm_slot(self):
        endpoint_key = self._llm_endpoint_key()
        semaphore = self._llm_bulkhead(endpoint_key, self.llm_max_inflight_per_endpoint)
        semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    def _recall_memory(self, question: str) -> tuple[str, int, str]:
        if self.memory_orchestrator is not None:
            result = self.memory_orchestrator.recall(
                question=question,
                user_id=self.user_id,
                memory_scope=self.memory_scope,
            )
            return result.text, result.count, result.status

        if not self.memory_tools:
            return 'No previous statements found.', 0, 'disabled'

        results = self.memory_tools.search_memories(
            query=question,
            user_id=self.user_id,
            filters={'memory_scope': self.memory_scope},
        )
        passages: list[str] = []
        status = 'empty'
        match results:
            case Success(memories):
                passages = memories.get('passages', [])
                status = 'retrieved' if passages else 'empty'
            case Failure(error):
                logger.error(f'[Agent]: Failed to search memories: {error}')
                passages = []
                status = 'error'
            case _:
                passages = []
                status = 'unknown'

        recalled_text = self.memory_tools.render_recalled_memory_text(passages)
        return recalled_text, len(passages), status

    def _store_memory(
        self,
        *,
        question: str,
        answer: str,
        reasoning: str,
        sample_id: str,
        round_index: str,
    ) -> None:
        if self.memory_orchestrator is not None:
            self.memory_orchestrator.store(
                question=question,
                answer=answer,
                reasoning=reasoning,
                user_id=self.user_id,
                metadata={
                    'agent': self.name,
                    'group': self.group,
                    'sample_id': sample_id,
                    'round_index': round_index,
                    'memory_scope': self.memory_scope,
                    'sample_run_id': self.run_id,
                },
            )
            return

        if not self.memory_tools:
            return

        memory_payload = self.memory_tools.format_store_memory_text(
            question=question,
            answer=answer,
            reasoning=reasoning,
        )

        store_result = self.memory_tools.store_memory(
            content=memory_payload,
            user_id=self.user_id,
            metadata={
                'agent': self.name,
                'group': self.group,
                'sample_id': sample_id,
                'round_index': round_index,
                'memory_scope': self.memory_scope,
                'sample_run_id': self.run_id,
            },
        )
        if isinstance(store_result, Failure):
            logger.warning(f'[Agent]: Failed to store memory: {store_result.failure()}')

    def forward(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str | None = None,
        update_instruction: str | None = None,
        logging_context: dict[str, Any] | None = None,
    ) -> Result[dspy.Prediction, AgentExecutionError]:
        """Genesis step when peer_answers is None, interaction update otherwise."""
        phase = self.lifecycle.transition_for_step(has_peer_answers=peer_answers is not None)
        context_payload = logging_context or {}
        sample_id = str(context_payload.get('sample_id', 'unknown'))
        round_index = str(context_payload.get('round_index', 'na'))
        artifact_root = str(context_payload.get('artifact_root', f'agent_logs/{self.run_id}'))
        phase_label = str(context_payload.get('phase', phase.value))
        recalled_memory_text = 'No previous statements found.'
        recalled_memory_count = 0
        recall_status = 'not_used'

        if phase is not AgentState.GENESIS:
            (
                recalled_memory_text,
                recalled_memory_count,
                recall_status,
            ) = self._recall_memory(question)

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
            'sample_id': sample_id,
            'round_index': round_index,
            'past_interaction_memory': recalled_memory_text,
            'past_interaction_memory_count': recalled_memory_count,
            'memory_recall_status': recall_status,
            'memory_scope': self.memory_scope,
        }
        if self.enable_runtime_artifacts:
            mlflow.log_dict(
                input_payload,
                f'{artifact_root}/round_{round_index}_{phase_label}/{self.name}_inputs.json',
            )

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
                    prediction_result = self._run_with_retry(
                        lambda: self._predict_update(
                            question=question,
                            context=context,
                            options=options,
                            system_prompt=system_prompt,
                            peer_answers=peer_answers or '',
                            update_instruction=update_instruction or '',
                            recalled_memory=recalled_memory_text,
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
        self._store_memory(
            question=question,
            answer=answer,
            reasoning=reasoning,
            sample_id=sample_id,
            round_index=round_index,
        )

        output_payload = {
            'answer': answer,
            'reasoning': reasoning,
            'agent_name': self.name,
            'sample_id': sample_id,
            'round_index': round_index,
            'phase': phase_label,
        }
        if self.enable_runtime_artifacts:
            mlflow.log_dict(
                output_payload,
                f'{artifact_root}/round_{round_index}_{phase_label}/{self.name}_output.json',
            )

        if hasattr(self.lm, 'last_token_usage'):
            mlflow.log_metric('tokens_used', getattr(self.lm, 'last_token_usage', 0))

        return Success(dspy.Prediction(answer=answer, reasoning=reasoning, agent_name=self.name))
