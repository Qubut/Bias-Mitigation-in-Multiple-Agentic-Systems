# bias_mitigation/mas/program.py
from __future__ import annotations

import uuid

import dspy
import mlflow
from mlflow.entities import SpanType

from bias_mitigation.data.models.config import InterventionType, MASConfig
from bias_mitigation.mas.protocols import ProtocolFactory
from bias_mitigation.memory.mem0_tools import Mem0Tools

from .agent import Agent
from .mas_statemachine import MASStateMachine


class MASProgram(dspy.Module):
    def __init__(self, config: MASConfig, memory_tools: Mem0Tools | None = None):
        super().__init__()
        self.config = config
        self.memory_tools = memory_tools

        if (
            config.intervention in {InterventionType.MEM0G, InterventionType.MEM0G_GEPA}
            and not memory_tools
        ):
            raise ValueError('Mem0Tools required for MEM0 interventions')

        self.protocol = ProtocolFactory.get(config.protocol)
        self.agent_lms = [self._build_lm(model) for model in config.agent_models]

    def _build_lm(self, agent_model):
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
            timeout=60,
        )
        lm.context_window_tokens_configured = int(agent_model.context_window_tokens)
        lm.max_tokens_configured = int(agent_model.max_tokens)
        return lm

    @mlflow.trace(name='MASProgram_Forward', span_type=SpanType.AGENT)
    def forward(
        self, context: str, question: str, ans0: str, ans1: str, ans2: str, category: str, **kwargs
    ) -> dspy.Prediction:
        """Execute one full MAS run for a single sample.

            The MAS lifecycle is managed by the MASStateMachine, which orchestrates
            agent interactions according to the specified protocol and collects
            per-agent prediction history for downstream metric computation.

            The output includes the full prediction history and final answers per
            agent, which are logged to MLflow for traceability and analysis.
            Returns:
                dspy.Prediction containing the full agent prediction history and
                final answers for the current sample.
        """
        base_groups = [g for g in (kwargs.get('stereotyped_groups') or []) if g] or [category]
        groups = [base_groups[i % len(base_groups)] for i in range(self.config.num_agents)]
        run_id = str(uuid.uuid4())

        agents = [
            Agent(
                name=self.config.agent_models[i % len(self.config.agent_models)].agent_name,
                group=groups[i],
                lm=self.agent_lms[i % len(self.agent_lms)],
                memory_tools=self.memory_tools,
                run_id=run_id,
            )
            for i in range(self.config.num_agents)
        ]
        options = [ans0, ans1, ans2]

        history = MASStateMachine(
            agents=agents,
            options=options,
            groups=groups,
            context=context,
            question=question,
            protocol=self.protocol,
            config=self.config,
        ).run()

        output_payload = {
            'history': {name: [p.answer for p in preds] for name, preds in history.items()},
            'final_answers': {name: preds[-1].answer for name, preds in history.items()},
            'entry_id': kwargs.get('id'),
        }
        mlflow.log_dict(output_payload, 'mas_full_output.json')
        mlflow.log_metric('mas.num_agents', self.config.num_agents)
        mlflow.log_metric('mas.rounds_completed', self.config.rounds)

        return dspy.Prediction(
            history=history,
            final_answers={name: preds[-1].answer for name, preds in history.items()},
        )
