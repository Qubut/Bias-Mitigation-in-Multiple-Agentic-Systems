from pathlib import Path

import yaml
from pydantic import AnyUrl, BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from returns.result import safe


class BBQConfig(BaseModel):
    """Configuration for BBQ dataset."""

    base_url: AnyUrl
    categories: list[str]
    dir_name: str = 'bbq'

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: AnyUrl) -> AnyUrl:
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


class MASConfig(BaseSettings):
    model_config = SettingsConfigDict(extra='forbid', env_prefix='MAS_')

    db_url: str = 'sqlite+aiosqlite:///./datasets.db'
    num_agents: int = 2
    rounds: int = 4
    protocol: str = 'cooperative'
    malicious: bool = False
    sample_size: int = 100
