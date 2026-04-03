"""Agent module for per-agent reasoning in MAS runs."""

import uuid
from typing import Any

import dspy
import mlflow
from mlflow.entities import SpanType

from .agent_statemachine import AgentState, AgentStateMachine
from .signatures import InitialAnswer, UpdateAnswer, UpdateAnswerWithMemory


class Agent(dspy.Module):
    """Per-agent reasoning module with optional memory-backed updates."""

    def __init__(
        self,
        name: str,
        group: str,
        lm: dspy.LM | None = None,
        memory_client: Any | None = None,
        run_id: str | None = None,
    ):
        """Initialize one agent runtime instance.

        Args:
            name: Agent identifier used in history and tracing.
            group: Social-group label used for prompt conditioning.
            lm: Optional DSPy language model instance.
            memory_client: Optional retrieval client for memory interventions.
            run_id: Optional run-scoped identifier for session isolation.
        """
        super().__init__()
        self.name = name
        self.group = group
        self.lm = lm
        # To strictly isolate memory per test case matching the preregistration,
        # we append a unique run_id to the user_id if provided.
        self.run_id = run_id or str(uuid.uuid4())
        self.user_id = f'{self.name}_{self.run_id}'
        self.lifecycle = AgentStateMachine()

        self.memory_client = memory_client

        with mlflow.start_span(name=f'Agent_Init_{name}', span_type=SpanType.AGENT) as span:
            span.set_attribute('agent.name', name)
            span.set_attribute('agent.group', group)
            span.set_attribute(
                'agent.lm_model', getattr(lm, 'model', 'unknown') if lm else 'default'
            )
            # NO log_params here — run-level params are immutable (see MASProgram_Init)

        # Predictor creation stays in __init__ (DSPy best practice)
        with dspy.context(lm=self.lm):
            self.initial = dspy.Predict(InitialAnswer)
            if self.memory_client:
                self.update = dspy.Predict(UpdateAnswerWithMemory)
            else:
                self.update = dspy.Predict(UpdateAnswer)

    @mlflow.trace(
        name='Agent_Forward',
        span_type=SpanType.AGENT,
        attributes={
            'agent.name': lambda self: self.name,
            'agent.group': lambda self: self.group,
        },
    )
    def forward(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str | None = None,
        update_instruction: str | None = None,
    ) -> dspy.Prediction:
        """Run one agent step for genesis or interaction phases.

        If ``peer_answers`` is absent, the agent produces a genesis response.
        Otherwise it performs an interaction update, optionally using recalled
        memory when a memory client is configured.
        """
        # Log structured input as artifact (reproducibility) — safe, unique filename
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
            if phase == AgentState.GENESIS:
                pred = self.initial(
                    question=question,
                    context=context,
                    options=options,
                    system_prompt=system_prompt,
                    group=self.group,
                )
            elif self.memory_client:
                mem_pred = self.memory_client(query_or_queries=question, user_id=self.user_id, k=5)
                recalled_memory = (
                    '\n'.join(mem_pred.passages)
                    if mem_pred.passages
                    else 'No previous statements found.'
                )

                pred = self.update(
                    question=question,
                    context=context,
                    options=options,
                    system_prompt=system_prompt,
                    peer_answers=peer_answers,
                    past_interaction_memory=recalled_memory,
                    protocol_instruction=update_instruction or '',
                    group=self.group,
                )
            else:
                pred = self.update(
                    question=question,
                    context=context,
                    options=options,
                    system_prompt=system_prompt,
                    peer_answers=peer_answers,
                    protocol_instruction=update_instruction or '',
                    group=self.group,
                )

            if self.memory_client:
                self.memory_client.bypass_inject(
                    messages=f'My previous answer: {pred.answer}. Reasoning: {pred.reasoning}',
                    user_id=self.user_id,
                    metadata={'agent': self.name},
                )

        output_payload = {
            'answer': pred.answer,
            'reasoning': pred.reasoning,
            'agent_name': self.name,
        }
        mlflow.log_dict(output_payload, f'agent_{self.name}_output.json')

        # Optional token usage metric (if LM exposes it)
        if hasattr(self.lm, 'last_token_usage'):
            mlflow.log_metric('tokens_used', getattr(self.lm, 'last_token_usage', 0))

        return dspy.Prediction(
            answer=pred.answer,
            reasoning=pred.reasoning,
            agent_name=self.name,
        )
