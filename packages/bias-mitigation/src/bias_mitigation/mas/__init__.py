"""MAS orchestration package.

Exports public runtime entry points for agent execution and orchestration.
Internal lifecycle and protocol modules remain available under submodules.
"""
from .agent import Agent
from .mas_program import MASProgram

__all__ = ['Agent', 'MASProgram']
