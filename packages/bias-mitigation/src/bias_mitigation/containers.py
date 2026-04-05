# bias_mitigation/containers.py
from dependency_injector import containers, providers

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.mas_program import MASProgram


class Container(containers.DeclarativeContainer):
    """Full enterprise DI container (official docs pattern)."""

    config = providers.Configuration()  # loads from dict/env/yaml

    mas_config = providers.Callable(MASConfig.model_validate, config.mas_config)

    # MASProgram factory (injects everything)
    mas_program = providers.Factory(
        MASProgram,
        config=mas_config,
    )
