"""Configuration datamodels and validation structures for the application."""

from enum import StrEnum
from importlib import import_module
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
        module = import_module('bias_mitigation.mas.intervention_strategy')
        return module.INTERVENTION_STRATEGY_MAP[self]


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
    agent_name: str


class MASConfig(BaseSettings):
    """Runtime configuration for MAS training and evaluation entry points."""

    model_config = SettingsConfigDict(extra='forbid', env_prefix='MAS_')
    db_url: str = 'sqlite+aiosqlite:///./datasets.db'
    num_agents: int = 2
    rounds: int = 4
    protocol: str = 'cooperative'
    malicious: bool = False
    sample_size: int = 100
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
