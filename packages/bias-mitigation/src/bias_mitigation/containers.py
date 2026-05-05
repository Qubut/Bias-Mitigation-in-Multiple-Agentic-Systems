# bias_mitigation/containers.py
from dependency_injector import containers, providers

from bias_mitigation.data.models.config import InterventionType, MASConfig
from bias_mitigation.mas.mas_program import MASProgram
from bias_mitigation.memory.mem0_tools import Mem0Tools
from bias_mitigation.memory.orchestration.models import (
    MemoryOrchestrationConfig as RuntimeMemoryOrchestrationConfig,
)
from bias_mitigation.memory.orchestration.service import MemoryOrchestrator


def _build_memory_tools(config: MASConfig) -> Mem0Tools | None:
    match config.intervention:
        case InterventionType.MEM0G | InterventionType.MEM0G_GEPA:
            if config.memory_config is None:
                raise ValueError(
                    'memory_config is required when intervention is mem0g or mem0g_gepa'
                )
            return Mem0Tools(config.memory_config)
        case _:
            return None


def _build_memory_orchestrator(config: MASConfig, tools: Mem0Tools | None) -> MemoryOrchestrator | None:
    if tools is None:
        return None

    if config.intervention not in {InterventionType.MEM0G, InterventionType.MEM0G_GEPA}:
        return None

    orchestration_config = RuntimeMemoryOrchestrationConfig(
        worker_threads=config.memory_orchestration.worker_threads,
        max_pending_store_tasks=config.memory_orchestration.max_pending_store_tasks,
        recall_timeout_ms=config.memory_orchestration.recall_timeout_ms,
        store_timeout_ms=config.memory_orchestration.store_timeout_ms,
        store_async=config.memory_orchestration.store_async,
        failure_trip_threshold=config.memory_orchestration.failure_trip_threshold,
        recovery_success_threshold=config.memory_orchestration.recovery_success_threshold,
    )
    return MemoryOrchestrator(memory_tools=tools, config=orchestration_config)


class Container(containers.DeclarativeContainer):
    """Full enterprise DI container (official docs pattern)."""

    config = providers.Configuration()  # loads from dict/env/yaml

    mas_config = providers.Singleton(MASConfig.model_validate, config.mas_config)
    memory_tools = providers.Singleton(_build_memory_tools, mas_config)
    memory_orchestrator = providers.Singleton(
        _build_memory_orchestrator,
        config=mas_config,
        tools=memory_tools,
    )

    mas_program = providers.Factory(
        MASProgram,
        config=mas_config,
        memory_tools=memory_tools,
        memory_orchestrator=memory_orchestrator,
    )
