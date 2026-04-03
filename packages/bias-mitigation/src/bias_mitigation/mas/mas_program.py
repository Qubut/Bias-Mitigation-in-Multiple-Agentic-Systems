"""Top-level MAS DSPy program orchestrating one evaluation sample."""

import threading
import uuid
from typing import Any

import dspy
import mlflow
from mlflow.entities import SpanType

from bias_mitigation.data.models.config import InterventionType, MASConfig
from bias_mitigation.mas.mas_statemachine import MASStateMachine
from bias_mitigation.mas.protocols import ProtocolFactory
from bias_mitigation.mas.tracing_manager import TracingManager
from bias_mitigation.memory import Mem0RM

from .agent import Agent


class MASProgram(dspy.Module):
    """DSPy module that runs genesis/interaction flow and returns final answers."""

    _autolog_enabled = False
    _observability_initialized = False
    _init_lock: threading.Lock | None = None  # lazy init for thread safety (imported at runtime)

    @classmethod
    def initialize_observability(cls):
        """Initialize global tracing/autolog integrations once per process."""
        TracingManager.initialize(enable_dspy_tracing=True)

    def __init__(self, config: MASConfig):
        """Create a program instance from runtime configuration.

        Args:
            config: Parsed MAS runtime configuration.

        Side Effects:
            - Configures retrieval model in ``dspy.settings`` when memory is enabled.
            - Constructs protocol and execution pools used by the state machine.
            - Logs initialization metadata to MLflow.
        """
        super().__init__()
        self.config = config

        self.memory_rm: Mem0RM | None = None

        # Strategy pattern based on intervention configuration
        self.intervention = getattr(config, 'intervention', InterventionType.BASELINE)
        match self.intervention:
            case InterventionType.MEM0G | InterventionType.MEM0G_GEPA:
                mem_cfg = getattr(config, 'memory_config', None)
                if not mem_cfg:
                    raise ValueError(
                        f'memory_config must be provided for intervention type {self.intervention}'
                    )
                # handle if memory_config is model instance by converting to dictionary
                if hasattr(mem_cfg, 'model_dump'):
                    mem_cfg_dict = mem_cfg.model_dump(exclude_none=True)
                else:
                    mem_cfg_dict = mem_cfg
                self.memory_rm = Mem0RM(mem_cfg_dict)
                dspy.settings.configure(rm=self.memory_rm)

        self.protocol = ProtocolFactory.get(config.protocol, config.malicious)

        self.genesis_executor = dspy.Parallel(
            num_threads=1,  # genesis is sequential (paper protocol)
            max_errors=0,
            disable_progress_bar=True,
            return_failed_examples=False,
        )
        self.update_executor = dspy.Parallel(
            num_threads=1,  # interaction rounds are sequential (paper protocol)
            max_errors=0,
            disable_progress_bar=True,
            return_failed_examples=False,
        )

        self.agent_lms = [
            dspy.LM(
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
            for agent_model in config.agent_models
        ]
        with mlflow.start_span(name='MASProgram_Init', span_type=SpanType.AGENT) as span:
            span.set_attribute('mas.num_agents', config.num_agents)
            span.set_attribute('mas.protocol', config.protocol)
            span.set_attribute('mas.malicious', config.malicious)
            mlflow.log_params({
                'num_agents': config.num_agents,
                'protocol': config.protocol,
                'malicious': config.malicious,
                'rounds': config.rounds,
            })

    @mlflow.trace(
        name='MASProgram_Forward',
        span_type=SpanType.AGENT,
        attributes={
            'mas.num_agents': lambda self: self.config.num_agents,
            'mas.protocol': lambda self: self.config.protocol,
        },
    )
    def forward(
        self,
        context: str,
        question: str,
        ans0: str,
        ans1: str,
        ans2: str,
        category: str,
        stereotyped_groups: list[str] | None = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        """Execute one full MAS run for a single sample.

        Returns:
            ``dspy.Prediction`` containing per-agent history and final answers.
        """
        groups = (stereotyped_groups[: self.config.num_agents] if stereotyped_groups else None) or [
            category
        ] * self.config.num_agents

        run_id = str(uuid.uuid4())

        agents = [
            Agent(
                name=self.config.agent_models[i % len(self.config.agent_models)].agent_name,
                group=groups[i % len(groups)],
                lm=self.agent_lms[i % len(self.agent_lms)],
                memory_client=self.memory_rm,
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
            genesis_executor=self.genesis_executor,
            update_executor=self.update_executor,
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
