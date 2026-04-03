"""Intervention strategy selection for MAS program construction."""

from abc import ABC, abstractmethod
from typing import Any

import dspy

from bias_mitigation.data.models.config import InterventionType
from bias_mitigation.mas.optimization import gepa_optimize_mas

from .mas_program import MASProgram


class InterventionStrategy(ABC):
    """Abstract strategy contract for intervention-specific program setup."""

    @abstractmethod
    def get_program(self, config: Any, trainset: list[dspy.Example] | None = None) -> MASProgram:
        """Return ready-to-evaluate program for this intervention."""


class BaselineStrategy(InterventionStrategy):
    """Construct the unoptimized baseline MAS program."""

    def get_program(self, config: Any, trainset: list[dspy.Example] | None = None) -> MASProgram:
        """Return a baseline ``MASProgram`` without optimization."""
        return MASProgram(config)


class BaselinePromptOptStrategy(InterventionStrategy):
    """Optimize baseline prompting with GEPA before evaluation."""

    def get_program(self, config: Any, trainset: list[dspy.Example] | None = None) -> MASProgram:
        """Return a GEPA-optimized baseline ``MASProgram``."""
        # GEPA on baseline (no memory)
        config.intervention = 'baseline'
        return gepa_optimize_mas(MASProgram(config), trainset)


class Mem0gStrategy(InterventionStrategy):
    """Construct a memory-enabled MAS program without GEPA."""

    def get_program(self, config: Any, trainset: list[dspy.Example] | None = None) -> MASProgram:
        """Return a ``MASProgram`` that uses configured memory retrieval."""
        return MASProgram(config)  # memory already enabled via config


class Mem0gGepaStrategy(InterventionStrategy):
    """Optimize memory-enabled prompting with GEPA and restore intervention tag."""

    def get_program(self, config: Any, trainset: list[dspy.Example] | None = None) -> MASProgram:
        """Return a GEPA-optimized memory-enabled ``MASProgram``."""
        # GEPA pre-optimization exactly as required by preregistration
        config.intervention = 'mem0g'  # optimize memory-aware signatures
        optimized = gepa_optimize_mas(MASProgram(config), trainset)
        # Re-apply memory after optimization
        optimized.intervention = 'mem0g_gepa'
        return optimized


INTERVENTION_STRATEGY_MAP: dict[InterventionType, InterventionStrategy] = {
    InterventionType.BASELINE: BaselineStrategy(),
    InterventionType.BASELINE_PROMPT_OPT: BaselinePromptOptStrategy(),
    InterventionType.MEM0G: Mem0gStrategy(),
    InterventionType.MEM0G_GEPA: Mem0gGepaStrategy(),
}
