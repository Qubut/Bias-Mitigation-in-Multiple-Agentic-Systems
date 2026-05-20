"""Typed Pydantic models for evaluator data contracts and artifacts."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _EvaluationMetadata(BaseModel):
    """Typed metadata used for stratification and evaluator artifacts."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    dataset_name: str
    dataset_source: str
    stereoset_type: str
    category: str
    protocol: str
    llm_models: str
    model_names: str
    intervention: str
    num_agents: str
    rounds: str
    split: str
    seed: str
    run_id: str
    agent_model_map: str


class _AgentTurnRow(BaseModel):
    """Typed per-turn artifact row for one agent step."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    sample_id: str
    sample_run_id: str | None = None
    example_index: int
    mlflow_run_id: str
    dataset_name: str
    dataset_source: str
    stereoset_type: str
    category: str
    protocol: str
    intervention: str
    agent_name: str
    agent_model_name: str
    turn_index: int
    phase: str
    answer: str
    reasoning: str
    gold_label: int
    gold_answer_text: str
    answer_matches_gold: bool
    is_biased_turn: bool
    is_final_turn: bool
    changed_from_previous: bool

    @classmethod
    def analysis_columns(cls) -> list[str]:
        return [name for name in cls.model_fields if name != 'sample_run_id']

    @classmethod
    def from_series(
        cls,
        *,
        base_turn_rows: list[dict[str, Any]],
        final_turn_index_by_agent: Mapping[str, int],
        sample_id: str,
        sample_run_id: str | None,
        example_index: int,
        metadata: _EvaluationMetadata,
    ) -> list['_AgentTurnRow']:
        typed_turn_rows = [_AgentTurnInput.model_validate(row) for row in base_turn_rows]
        previous_answer_by_agent: dict[str, str] = {}

        rows: list[_AgentTurnRow] = []
        for turn in typed_turn_rows:
            previous_answer = previous_answer_by_agent.get(turn.agent_name)
            rows.append(
                cls(
                    sample_id=sample_id,
                    sample_run_id=sample_run_id,
                    example_index=example_index,
                    mlflow_run_id=metadata.run_id,
                    dataset_name=metadata.dataset_name,
                    dataset_source=metadata.dataset_source,
                    stereoset_type=metadata.stereoset_type,
                    category=metadata.category,
                    protocol=metadata.protocol,
                    intervention=metadata.intervention,
                    agent_name=turn.agent_name,
                    agent_model_name=turn.agent_model_name,
                    turn_index=turn.turn_index,
                    phase=turn.phase,
                    answer=turn.answer_text,
                    reasoning=turn.reasoning_text,
                    gold_label=turn.gold_label,
                    gold_answer_text=turn.gold_answer_text,
                    answer_matches_gold=turn.answer_matches_gold,
                    is_biased_turn=turn.is_biased_turn,
                    is_final_turn=turn.turn_index
                    == final_turn_index_by_agent.get(turn.agent_name, 0),
                    changed_from_previous=previous_answer is not None
                    and previous_answer != turn.answer_text,
                )
            )
            previous_answer_by_agent[turn.agent_name] = turn.answer_text

        return rows


class _AgentTurnInput(BaseModel):
    """Validated source payload for one per-agent turn row."""

    model_config = ConfigDict(extra='ignore', frozen=True)

    agent_name: str
    agent_model_name: str
    turn_index: int
    phase: str
    answer_text: str
    reasoning_text: str
    gold_label: int
    gold_answer_text: str
    answer_matches_gold: bool
    is_biased_turn: bool


class _AgentBiasSummary(BaseModel):
    """Derived bias summary maps computed from agent turn rows."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    first_biased_turn_by_agent: dict[str, int] = Field(default_factory=dict)
    final_is_biased_by_agent: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def from_turn_rows(cls, agent_turn_rows: list[_AgentTurnRow]) -> '_AgentBiasSummary':
        sorted_rows = sorted(agent_turn_rows, key=lambda item: (item.agent_name, item.turn_index))
        rows_by_agent: dict[str, list[_AgentTurnRow]] = {}
        for row in sorted_rows:
            rows_by_agent.setdefault(row.agent_name, []).append(row)

        first_biased_turn_by_agent = {
            agent_name: first_biased_row.turn_index
            for agent_name, rows in rows_by_agent.items()
            if (
                first_biased_row := next((row for row in rows if row.is_biased_turn), None)
            )
            is not None
        }
        final_is_biased_by_agent = {
            agent_name: final_row.is_biased_turn
            for agent_name, rows in rows_by_agent.items()
            if (
                final_row := next((row for row in reversed(rows) if row.is_final_turn), None)
            )
            is not None
        }

        return cls(
            first_biased_turn_by_agent=first_biased_turn_by_agent,
            final_is_biased_by_agent=final_is_biased_by_agent,
        )


class _RoundMetricRow(BaseModel):
    """Typed per-round metrics row for one evaluated sample."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    sample_id: str
    example_index: int
    mlflow_run_id: str
    dataset_name: str
    dataset_source: str
    stereoset_type: str
    category: str
    protocol: str
    intervention: str
    model_names: str
    turn_index: int
    robustness_rate: float
    bias_prevalence: float
    propagation_pr_t: float
    first_biased_turn: int
    emergence_observed: bool
    biased_agent_count: int = 0
    biased_agents: list[str] = Field(default_factory=list)
    biased_models: list[str] = Field(default_factory=list)
    agent_bias_flags: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def analysis_columns(cls) -> list[str]:
        return list(cls.model_fields.keys())

    @classmethod
    def from_components(
        cls,
        *,
        sample_id: str,
        example_index: int,
        metadata: _EvaluationMetadata,
        round_metrics: Mapping[str, Any],
        bias_row: Mapping[str, Any] | None = None,
    ) -> '_RoundMetricRow':
        typed_round_metrics = _RoundMetricInput.model_validate(round_metrics)
        typed_bias_row = _RoundBiasInput.model_validate(bias_row or {})

        return cls(
            sample_id=sample_id,
            example_index=example_index,
            mlflow_run_id=metadata.run_id,
            dataset_name=metadata.dataset_name,
            dataset_source=metadata.dataset_source,
            stereoset_type=metadata.stereoset_type,
            category=metadata.category,
            protocol=metadata.protocol,
            intervention=metadata.intervention,
            model_names=metadata.model_names,
            turn_index=typed_round_metrics.turn_index,
            robustness_rate=typed_round_metrics.robustness_rate,
            bias_prevalence=typed_round_metrics.bias_prevalence,
            propagation_pr_t=typed_round_metrics.propagation_pr_t,
            first_biased_turn=typed_round_metrics.first_biased_turn,
            emergence_observed=typed_round_metrics.emergence_observed,
            biased_agent_count=typed_bias_row.biased_agent_count,
            biased_agents=typed_bias_row.biased_agents,
            biased_models=typed_bias_row.biased_models,
            agent_bias_flags=typed_bias_row.agent_bias_flags,
        )


class _RoundMetricInput(BaseModel):
    """Validated source payload for one computed round metric row."""

    model_config = ConfigDict(extra='ignore', frozen=True)

    turn_index: int
    robustness_rate: float
    bias_prevalence: float
    propagation_pr_t: float
    first_biased_turn: int
    emergence_observed: bool


class _RoundBiasInput(BaseModel):
    """Validated source payload for round bias attribution metadata."""

    model_config = ConfigDict(extra='ignore', frozen=True)

    biased_agent_count: int = 0
    biased_agents: list[str] = Field(default_factory=list)
    biased_models: list[str] = Field(default_factory=list)
    agent_bias_flags: dict[str, bool] = Field(default_factory=dict)


class _SampleOutcomeRow(BaseModel):
    """Typed sample-level evaluation artifact row."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    sample_id: str
    example_index: int
    mlflow_run_id: str
    dataset_name: str
    dataset_source: str
    stereoset_type: str
    category: str
    protocol: str
    llm_models: str
    model_names: str
    intervention: str
    num_agents: str
    rounds: str
    split: str
    seed: str
    question_polarity: str
    context_condition: str
    label: Any
    gold_answer_text: str
    system_robustness: float
    emergence_rate: float
    amplification_rate: float
    propagation_rate: float
    turn_count: int
    processed_flag: bool
    failure_reason: str | None
    sample_run_id: str | None = None
    agent_model_map: str = '{}'
    first_biased_turn_by_agent: dict[str, int] = Field(default_factory=dict)
    final_is_biased_by_agent: dict[str, bool] = Field(default_factory=dict)
    final_answers: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def analysis_columns(cls) -> list[str]:
        return list(cls.model_fields.keys())


class _FailureExampleRow(BaseModel):
    """Typed failure row emitted for unsuccessful deterministic tasks."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    index: int
    error: str
    sample_id: str
    metadata: _EvaluationMetadata


class _StratifiedRow(BaseModel):
    """Typed stratified aggregate row."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    dimensions: dict[str, str]
    support: int
    metrics: dict[str, float]
    ci95: dict[str, float]


class _EvaluatorOutput(BaseModel):
    """Typed evaluator output contract."""

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    system_robustness: float
    detailed_results: list[dict[str, Any]] | None
    config: Any
    overall_metrics: dict[str, float]
    stratified_metrics: list[dict[str, Any]]
    uncertainty: dict[str, float]
    failure_count: int
    processed_count: int
    failed_examples: list[_FailureExampleRow]
    sample_outcomes: list[_SampleOutcomeRow]
    agent_turns: list[_AgentTurnRow]
    analysis_schema: dict[str, Any]
    stream_metric_rows: list[dict[str, Any]]
    stream_round_metric_rows: list[dict[str, Any]]
    stream_failure_rows: list[dict[str, Any]]
    stream_summary: dict[str, int]
