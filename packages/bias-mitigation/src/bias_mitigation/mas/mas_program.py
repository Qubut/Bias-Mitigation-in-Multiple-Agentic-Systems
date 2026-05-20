"""dspy.Module for one sample. Owns the predictors GEPA optimises."""

from __future__ import annotations

import copy
import uuid
from hashlib import sha256
from typing import Any
from typing import cast as _cast

import dspy
import mlflow

from bias_mitigation.data.models.config import InterventionType, MASConfig
from bias_mitigation.mas.protocols import ProtocolFactory
from bias_mitigation.memory.mem0_tools import Mem0Tools
from bias_mitigation.memory.orchestration.service import MemoryOrchestrator

from .agent import Agent
from .mas_statemachine import MASStateMachine
from .signatures import InitialAnswer, UpdateAnswer, UpdateAnswerWithMemory

# Don't deepcopy live resources (HTTP sessions, breakers, vector stores) —
# GEPA's per-candidate copy will fall back to shallow and warn at us.
_SHARED_BY_REFERENCE_ATTRS: tuple[str, ...] = (
    'memory_tools',
    'memory_orchestrator',
    'agent_lms',
)


class MASProgram(dspy.Module):
    """One MAS debate per aforward.

    initial + update are the GEPA search variables. Memory, retries, and
    concurrency are config knobs.
    """

    def __init__(
        self,
        config: MASConfig,
        memory_tools: Mem0Tools | None = None,
        memory_orchestrator: MemoryOrchestrator | None = None,
    ):
        super().__init__()
        self.config = config
        self.memory_tools = memory_tools
        self.memory_orchestrator = memory_orchestrator

        if (
            config.intervention in {InterventionType.MEM0G, InterventionType.MEM0G_GEPA}
            and not memory_tools
        ):
            raise ValueError('Mem0Tools required for MEM0 interventions')

        self.protocol = ProtocolFactory.get(config.protocol)
        self.agent_lms = [self._build_lm(model) for model in config.agent_models]

        memory_enabled = memory_tools is not None or memory_orchestrator is not None
        self.initial = dspy.Predict(InitialAnswer)
        self.update = dspy.Predict(UpdateAnswerWithMemory if memory_enabled else UpdateAnswer)

    def __deepcopy__(self, memo: dict[int, Any]) -> MASProgram:
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in _SHARED_BY_REFERENCE_ATTRS:
                new.__dict__[k] = v
            else:
                new.__dict__[k] = copy.deepcopy(v, memo)
        return new

    @classmethod
    def syncified(
        cls,
        config: MASConfig,
        memory_tools: Mem0Tools | None = None,
        memory_orchestrator: MemoryOrchestrator | None = None,
    ) -> MASProgram:
        """Construct and dspy.syncify in place. Use this from DI."""
        program = cls(
            config=config,
            memory_tools=memory_tools,
            memory_orchestrator=memory_orchestrator,
        )
        return _cast('MASProgram', dspy.syncify(program, in_place=True))

    def _build_lm(self, agent_model) -> dspy.LM:
        lm = dspy.LM(
            model=agent_model.name,
            api_key=agent_model.api_key.get_secret_value()
            if hasattr(agent_model.api_key, 'get_secret_value')
            else agent_model.api_key,
            api_base=agent_model.api_base,
            cache=False,
            model_type='chat',
            temperature=agent_model.temperature,
            max_tokens=agent_model.max_tokens,
            num_retries=3,
            timeout=self.config.evaluator_concurrency.llm_timeout_seconds,
        )
        lm.context_window_tokens_configured = int(agent_model.context_window_tokens)
        lm.max_tokens_configured = int(agent_model.max_tokens)
        return lm

    async def aforward(
        self,
        context: str,
        question: str,
        ans0: str,
        ans1: str,
        ans2: str,
        category: str,
        **kwargs,
    ) -> dspy.Prediction:
        """One debate.

        Returns Prediction(history, final_answers, sample_id,
        sample_run_id, memory_scope). kwargs: stereotyped_groups,
        sample_id|id.
        """
        base_groups = [g for g in (kwargs.get('stereotyped_groups') or []) if g] or [category]
        groups = [base_groups[i % len(base_groups)] for i in range(self.config.num_agents)]
        run_id = str(uuid.uuid4())
        active_run = mlflow.active_run()
        experiment_run_id = (
            active_run.info.run_id
            if active_run is not None and active_run.info is not None
            else 'unknown'
        )
        memory_scope = f'experiment:{experiment_run_id}'
        sample_id_input = kwargs.get('sample_id') or kwargs.get('id')
        if isinstance(sample_id_input, (str, int)) and str(sample_id_input).strip():
            sample_id = str(sample_id_input).strip()
        else:
            digest = sha256(f'{context}\n{question}\n{category}'.encode()).hexdigest()[:16]
            sample_id = f'sample-{digest}'

        llm_inflight_budget = max(
            1,
            int(self.config.evaluator_concurrency.max_llm_inflight_per_endpoint),
        )
        agents = [
            Agent(
                name=self.config.agent_models[i % len(self.config.agent_models)].agent_name,
                group=groups[i],
                lm=self.agent_lms[i % len(self.agent_lms)],
                initial=self.initial,
                update=self.update,
                memory_tools=self.memory_tools,
                memory_orchestrator=self.memory_orchestrator,
                run_id=run_id,
                enable_runtime_artifacts=False,
                llm_max_inflight_per_endpoint=llm_inflight_budget,
                llm_retry_attempts=self.config.evaluator_concurrency.llm_retry_attempts,
                llm_retry_backoff_min_seconds=self.config.evaluator_concurrency.llm_retry_backoff_min_seconds,
                llm_retry_backoff_max_seconds=self.config.evaluator_concurrency.llm_retry_backoff_max_seconds,
            )
            for i in range(self.config.num_agents)
        ]
        sm = MASStateMachine(
            agents=agents,
            options=[ans0, ans1, ans2],
            groups=groups,
            context=context,
            question=question,
            protocol=self.protocol,
            config=self.config,
            sample_id=sample_id,
            run_id=run_id,
        )
        await sm.activate_initial_state()
        history = sm.history

        return dspy.Prediction(
            history=history,
            final_answers={name: preds[-1].answer for name, preds in history.items()},
            sample_id=sample_id,
            sample_run_id=run_id,
            memory_scope=memory_scope,
        )
