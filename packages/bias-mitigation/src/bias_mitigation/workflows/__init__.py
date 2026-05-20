"""Public API for the bias-mitigation declarative workflow runtime.

This subpackage implements the Pipeline pattern that drives both evaluation
and training runs of the multi-agent system (MAS). Each run is modelled as a
deterministic state machine that walks through the canonical stages
``prepare`` -> ``build`` -> ``execute`` -> ``persist``, falling back to
``fail`` on any unrecoverable error. The shared ``RunContext`` dataclass is
threaded through every stage so the stages stay decoupled and individually
testable.

The module re-exports the two concrete runtimes (evaluation and training),
their builders, the ``WorkflowMachine`` state chart that drives them, and
the ``RunRequest`` / ``RunContext`` / ``RunMode`` contracts used by CLI
entrypoints such as ``scripts/evaluate.py`` and ``scripts/train.py``.
"""

from .contracts import RunContext, RunMode, RunRequest
from .evaluation import EvaluationWorkflowRuntimeImpl, build_evaluation_workflow_runtime
from .statechart import WorkflowMachine
from .training import TrainingWorkflowRuntimeImpl, build_training_workflow_runtime

__all__ = [
    'EvaluationWorkflowRuntimeImpl',
    'RunContext',
    'RunMode',
    'RunRequest',
    'TrainingWorkflowRuntimeImpl',
    'WorkflowMachine',
    'build_evaluation_workflow_runtime',
    'build_training_workflow_runtime',
]
