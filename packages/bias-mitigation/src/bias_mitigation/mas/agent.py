"""Per-agent DSPy reasoning module used by the MAS evaluation pipeline.

``Agent`` is a thin orchestration shell that pairs a pair of shared DSPy
predictors (owned by :class:`MASProgram`) with two side concerns:

* the lifecycle + per-phase predictor dispatch + bulkheaded retry, all
  delegated to :class:`AgentStateMachine`, and
* the surrounding I/O — Mem0 recall/store and MLflow artefact logging.

Memory and predictor calls are all driven asynchronously; the sync DSPy
predictor is shipped to :func:`asyncio.to_thread` inside
:meth:`AgentStateMachine.run_predictor`.
"""

import uuid
from typing import Any

import dspy
import mlflow
from loguru import logger
from returns.result import Failure, Result, Success

from bias_mitigation.mas.agent_statemachine import (
    AgentExecutionError,
    AgentState,
    AgentStateMachine,
    RetryConfig,
)
from bias_mitigation.memory import Mem0Tools
from bias_mitigation.memory.orchestration.service import MemoryOrchestrator


class Agent(dspy.Module):
    """One MAS role: shared predictors + memory I/O + lifecycle delegation."""

    def __init__(
        self,
        name: str,
        group: str,
        lm: dspy.LM,
        initial: dspy.Predict,
        update: dspy.Predict,
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
            active_run.info.run_id
            if active_run is not None and active_run.info is not None
            else self.run_id
        )
        self.memory_scope = f'experiment:{experiment_run_id}'
        self.user_id = f'{self.name}_{self.run_id}'
        self.lifecycle = AgentStateMachine()
        self.enable_runtime_artifacts = enable_runtime_artifacts
        backoff_min = max(0.0, llm_retry_backoff_min_seconds)
        self.retry_config = RetryConfig(
            attempts=max(1, llm_retry_attempts),
            backoff_min_seconds=backoff_min,
            backoff_max_seconds=max(backoff_min, llm_retry_backoff_max_seconds),
            max_inflight_per_endpoint=max(1, llm_max_inflight_per_endpoint),
        )
        # Shared predictors owned by MASProgram (GEPA visibility).
        self.initial = initial
        self.update = update

    def _endpoint_key(self) -> str:
        model = str(getattr(self.lm, 'model', 'unknown-model'))
        base = str(getattr(self.lm, 'api_base', 'unknown-base'))
        return f'{base}|{model}'

    @staticmethod
    def _normalize_reasoning(raw: object) -> str:
        if isinstance(raw, str) and raw.strip():
            return raw
        return 'No reasoning provided.'

    async def _recall_memory(self, question: str) -> tuple[str, int, str]:
        """Return ``(rendered_text, count, status)``; degrades to empty on any backend hiccup."""
        if self.memory_orchestrator is not None:
            result = await self.memory_orchestrator.recall(
                question=question,
                user_id=self.user_id,
                memory_scope=self.memory_scope,
            )
            return result.text, result.count, result.status

        if not self.memory_tools:
            return 'No previous statements found.', 0, 'disabled'

        results = await self.memory_tools.search_memories(
            query=question,
            user_id=self.user_id,
            filters={'memory_scope': self.memory_scope},
        )
        match results:
            case Success(memories):
                passages = memories.get('passages', [])
                status = 'retrieved' if passages else 'empty'
            case Failure(error):
                logger.error(f'[Agent]: Failed to search memories: {error}')
                passages, status = [], 'error'
            case _:
                passages, status = [], 'unknown'
        return (
            self.memory_tools.render_recalled_memory_text(passages),
            len(passages),
            status,
        )

    async def _store_memory(
        self,
        *,
        question: str,
        answer: str,
        reasoning: str,
        sample_id: str,
        round_index: str,
    ) -> None:
        """Best-effort persist of this turn; backend errors logged, never propagated."""
        metadata = {
            'agent': self.name,
            'group': self.group,
            'sample_id': sample_id,
            'round_index': round_index,
            'memory_scope': self.memory_scope,
            'sample_run_id': self.run_id,
        }
        if self.memory_orchestrator is not None:
            await self.memory_orchestrator.store(
                question=question,
                answer=answer,
                reasoning=reasoning,
                user_id=self.user_id,
                metadata=metadata,
            )
            return
        if not self.memory_tools:
            return

        payload = self.memory_tools.format_store_memory_text(
            question=question,
            answer=answer,
            reasoning=reasoning,
        )
        store_result = await self.memory_tools.store_memory(
            content=payload,
            user_id=self.user_id,
            metadata=metadata,
        )
        if isinstance(store_result, Failure):
            logger.warning(f'[Agent]: Failed to store memory: {store_result.failure()}')

    def _log_artifact(self, path: str, payload: dict[str, Any]) -> None:
        if self.enable_runtime_artifacts:
            mlflow.log_dict(payload, path)

    async def aforward(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str | None = None,
        update_instruction: str | None = None,
        logging_context: dict[str, Any] | None = None,
    ) -> Result[dspy.Prediction, AgentExecutionError]:
        """Run one deliberation step (genesis if no peer answers, otherwise peer-aware update)."""
        phase = self.lifecycle.transition_for_step(has_peer_answers=peer_answers is not None)
        ctx = logging_context or {}
        sample_id = str(ctx.get('sample_id', 'unknown'))
        round_index = str(ctx.get('round_index', 'na'))
        artifact_root = f'{ctx.get("artifact_root", f"agent_logs/{self.run_id}")}/round_{round_index}_{ctx.get("phase", phase.value)}'

        recalled_text, recalled_count, recall_status = (
            ('No previous statements found.', 0, 'not_used')
            if phase is AgentState.GENESIS
            else await self._recall_memory(question)
        )

        self._log_artifact(
            f'{artifact_root}/{self.name}_inputs.json',
            {
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
                'past_interaction_memory': recalled_text,
                'past_interaction_memory_count': recalled_count,
                'memory_recall_status': recall_status,
                'memory_scope': self.memory_scope,
            },
        )

        predictor_call = AgentStateMachine.predictor_call(
            phase,
            initial=self.initial,
            update=self.update,
            group=self.group,
            question=question,
            context=context,
            options=options,
            system_prompt=system_prompt,
            peer_answers=peer_answers,
            update_instruction=update_instruction,
            recalled_memory=recalled_text,
        )
        with dspy.context(lm=self.lm):
            prediction_result = await AgentStateMachine.run_predictor(
                predictor_call,
                phase=phase,
                agent_name=self.name,
                endpoint_key=self._endpoint_key(),
                retry_config=self.retry_config,
            )

        if isinstance(prediction_result, Failure):
            err = prediction_result.failure()
            logger.error(
                f'[Agent]: Prediction failed after retries for {err.agent_name} '
                f'(phase={err.phase}): {err.reason}'
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
        await self._store_memory(
            question=question,
            answer=answer,
            reasoning=reasoning,
            sample_id=sample_id,
            round_index=round_index,
        )
        self._log_artifact(
            f'{artifact_root}/{self.name}_output.json',
            {
                'answer': answer,
                'reasoning': reasoning,
                'agent_name': self.name,
                'sample_id': sample_id,
                'round_index': round_index,
                'phase': str(ctx.get('phase', phase.value)),
            },
        )
        if hasattr(self.lm, 'last_token_usage'):
            mlflow.log_metric('tokens_used', getattr(self.lm, 'last_token_usage', 0))
        return Success(dspy.Prediction(answer=answer, reasoning=reasoning, agent_name=self.name))


__all__ = ['Agent', 'AgentExecutionError']
