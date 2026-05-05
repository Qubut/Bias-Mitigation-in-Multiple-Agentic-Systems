"""Configuration datamodels and validation structures for the application."""

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from omegaconf import OmegaConf
from pydantic import AnyUrl, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from returns.result import safe

from bias_mitigation.data.models.memory_config import Mem0Config

if TYPE_CHECKING:
    from bias_mitigation.mas.intervention_strategy import InterventionStrategy


class InterventionType(StrEnum):
    """Enumeration of supported intervention strategies for bias mitigation."""

    BASELINE = 'baseline'
    BASELINE_PROMPT_OPT = 'baseline_prompt_opt'
    MEM0G = 'mem0g'
    MEM0G_GEPA = 'mem0g_gepa'

    def to_strategy(self) -> 'InterventionStrategy':
        """Resolve this intervention enum value to its strategy implementation."""
        from bias_mitigation.mas.intervention_strategy import (
            INTERVENTION_STRATEGY_MAP,
            InterventionStrategy as RuntimeInterventionStrategy,
        )

        strategy: RuntimeInterventionStrategy = INTERVENTION_STRATEGY_MAP[self]
        return strategy


class BBQConfig(BaseModel):
    """Configuration for BBQ dataset."""

    base_url: AnyUrl
    categories: list[str]
    dir_name: str = 'bbq'

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: AnyUrl) -> AnyUrl:
        """Ensure the base URL ends with a trailing slash."""
        if not str(v).endswith('/'):
            raise ValueError("BBQ base_url must end with '/'")
        return v


class StereoSetConfig(BaseModel):
    """Configuration for StereoSet dataset."""

    files: dict[str, AnyUrl]
    dir_name: str = 'stereoset'


class AppConfig(BaseSettings):
    """Root application configuration."""

    bbq: BBQConfig
    stereoset: StereoSetConfig

    model_config = SettingsConfigDict(
        env_prefix='DOWNLOAD_',
        env_nested_delimiter='__',
        extra='forbid',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prioritize init over env variables for Settings parsing."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    @safe(exceptions=(FileNotFoundError, yaml.YAMLError, ValueError))
    def from_yaml(cls, path: Path) -> 'AppConfig':
        """Static Factory method mapping raw YAML safely to domain config schemas."""
        with path.open(encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


class AgentModelConfig(BaseModel):
    """Configuration for individual agent models."""

    name: str
    api_key: SecretStr
    api_base: str
    temperature: float = 0.0
    max_tokens: int = 2048
    context_window_tokens: int = 16384
    agent_name: str


class EvaluatorConcurrencyConfig(BaseModel):
    """Concurrency controls grouped under a single evaluator config section."""

    max_evaluation_threads: int = Field(
        default=8,
        ge=1,
        description='Hard cap for evaluator execution threads across deterministic modes.',
    )
    max_llm_inflight_per_endpoint: int = Field(
        default=3,
        ge=1,
        description='Maximum concurrent LLM calls per endpoint (api_base + model).',
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description='Per-attempt network timeout for LLM calls.',
    )
    llm_retry_attempts: int = Field(
        default=3,
        ge=1,
        description='Maximum retry attempts per LLM call.',
    )
    llm_retry_backoff_min_seconds: float = Field(
        default=1.0,
        ge=0,
        description='Minimum exponential backoff delay between LLM retries.',
    )
    llm_retry_backoff_max_seconds: float = Field(
        default=4.0,
        ge=0,
        description='Maximum exponential backoff delay between LLM retries.',
    )
    enable_monitoring: bool = Field(
        default=False,
        description='Emit concurrency manager monitoring logs during deterministic evaluation.',
    )


class MemoryOrchestrationConfig(BaseModel):
    """Queue/worker orchestration settings for Mem0 runtime access."""

    worker_threads: int = Field(default=4, ge=1)
    max_pending_store_tasks: int = Field(default=128, ge=1)
    recall_timeout_ms: int = Field(default=6000, ge=100)
    store_timeout_ms: int = Field(default=2500, ge=100)
    store_async: bool = True
    failure_trip_threshold: int = Field(default=8, ge=1)
    recovery_success_threshold: int = Field(default=6, ge=1)


class MASConfig(BaseSettings):
    """Runtime configuration for MAS training and evaluation entry points."""

    model_config = SettingsConfigDict(extra='forbid', env_prefix='MAS_')
    db_url: str = 'sqlite+aiosqlite:///./datasets.db'
    num_agents: int = 2
    rounds: int = 4
    protocol: str = 'cooperative'
    malicious: bool = False
    sample_size: int = 100
    evaluator_max_errors: int | None = Field(
        default=None,
        ge=1,
        description='Maximum parallel worker errors before dspy.Parallel aborts.',
    )
    evaluator_disable_progress_bar: bool = Field(
        default=False,
        description='Disable dspy.Parallel progress bar for deterministic evaluation.',
    )
    evaluator_deterministic_execution_mode: str = Field(
        default='sliding_window',
        description='Deterministic execution mode: sliding_window or dspy_batch.',
    )
    evaluator_window_size: int | None = Field(
        default=None,
        ge=1,
        description='Optional fixed sliding-window size for deterministic execution.',
    )
    evaluator_task_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description='Per-sample timeout in seconds for deterministic sliding-window execution.',
    )
    evaluator_concurrency: EvaluatorConcurrencyConfig = Field(
        default_factory=EvaluatorConcurrencyConfig,
        description='Grouped concurrency controls for deterministic evaluator execution.',
    )
    analysis_artifact_root: str = Field(
        default='evaluation/analysis/v1',
        description='MLflow artifact root for analysis exports.',
    )
    analysis_local_root: str = Field(
        default='evaluation/analysis/live',
        description='Local directory root for live streaming analysis outputs.',
    )
    stream_flush_every_events: int = Field(
        default=1,
        ge=1,
        description='Flush local live stream files after this many emitted events.',
    )
    stream_fsync: bool = Field(
        default=False,
        description='If true, fsync local stream files for stronger durability.',
    )
    stream_live_csv: bool = Field(
        default=True,
        description='If true, write live CSV mirrors for stream rows.',
    )
    stream_max_buffered_events: int = Field(
        default=2048,
        ge=1,
        description='Maximum buffered stream events before applying backpressure policy.',
    )
    stream_drop_events_on_backpressure: bool = Field(
        default=False,
        description='If true, drop stream events when buffer is full instead of blocking emitters.',
    )
    analysis_live_dir_template: str = Field(
        default='{started_at}_{run_name}_{intervention}_{run_id_short}',
        description='Template used to build readable live analysis directory names.',
    )
    analysis_live_slug_max_length: int = Field(
        default=48,
        ge=8,
        description='Maximum slug length for each token used in live directory names.',
    )
    analysis_live_write_manifest: bool = Field(
        default=True,
        description='If true, write run manifest metadata inside each live analysis directory.',
    )
    analysis_live_index_filename: str = Field(
        default='runs_index.jsonl',
        description='Root-level JSONL index file mapping live directories to run metadata.',
    )
    reset_memory_on_genesis: bool = Field(
        default=False,
        description='If true, state machine clears session-scoped memory at genesis start.',
    )
    agent_models: list[AgentModelConfig] = Field(
        description='List of model names, where each agent gets its own independent LLM.'
    )
    intervention: str = Field(
        default='baseline',
        description="Must be one of: 'baseline', 'baseline_prompt_opt', 'mem0g', 'mem0g_gepa'",
    )
    memory_config: Mem0Config | None = Field(
        default=None,
        description='Configuration dictionary for Mem0 if intervention includes memory.',
    )
    memory_orchestration: MemoryOrchestrationConfig = Field(
        default_factory=MemoryOrchestrationConfig,
        description='State-machine orchestration and worker limits for Mem0 operations.',
    )

    @classmethod
    @safe(exceptions=(FileNotFoundError, yaml.YAMLError, ValueError))
    def load(cls, yaml_path: str = 'config.yaml') -> 'MASConfig':
        """Deserialise a YAML file into a validated configuration.

        Args:
            yaml_path: Filesystem path to the configuration YAML.

        Returns:
            A fully validated configuration instance.
        """
        cfg = OmegaConf.load(yaml_path)
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    @safe(exceptions=(FileNotFoundError, yaml.YAMLError, ValueError))
    def load_merged(
        cls,
        base_path: str,
        *override_paths: str,
        cli_overrides: dict[str, Any] | None = None,
    ) -> 'MASConfig':
        """Load and merge multiple YAML files with optional CLI overrides.

        Files are merged in order: base_path, then each override_path,
        then cli_overrides. Later values override earlier ones.

        Args:
            base_path: Path to the base configuration YAML.
            *override_paths: Additional YAML files to merge (in order).
            cli_overrides: Optional dict of CLI overrides (dot-notation keys).

        Returns:
            A fully validated, merged configuration instance.

        Example::

            config = MASConfig.load_merged(
                'config/base.yaml',
                'config/local.yaml',
                cli_overrides={'intervention': 'mem0g', 'agent_models[0].temperature': 0.7},
            )
        """
        base = OmegaConf.load(base_path)

        for path in override_paths:
            override = OmegaConf.load(path)
            base = OmegaConf.merge(base, override)

        if cli_overrides:
            cli_cfg = OmegaConf.create(cli_overrides)
            base = OmegaConf.merge(base, cli_cfg)

        return cls.model_validate(OmegaConf.to_container(base, resolve=True))

    @field_validator('memory_config')
    @classmethod
    def validate_memory_for_intervention(cls, v, info):
        """Require a valid memory configuration for memory-based interventions."""
        intervention = info.data.get('intervention')
        if intervention in {InterventionType.MEM0G, InterventionType.MEM0G_GEPA} and not v:
            raise ValueError('memory_config required for Mem0g interventions')
        return v
