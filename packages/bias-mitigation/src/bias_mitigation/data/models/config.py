from pydantic import AnyUrl, BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class BBQConfig(BaseModel):
    """Configuration for BBQ dataset."""

    base_url: AnyUrl
    categories: list[str]

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: AnyUrl) -> AnyUrl:
        if not str(v).endswith('/'):
            raise ValueError("BBQ base_url must end with '/'")
        return v


class StereoSetConfig(BaseModel):
    """Configuration for StereoSet dataset."""

    files: dict[str, AnyUrl]


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
