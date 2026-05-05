"""Declarative workflow statechart used by evaluate/train entrypoints."""

from __future__ import annotations

from typing import Protocol

from statemachine import State, StateChart

from .contracts import RunContext


class WorkflowRuntime(Protocol):
    """Typed runtime contract for workflow state transitions."""

    def prepare(self, context: RunContext) -> RunContext: ...

    def build(self, context: RunContext) -> RunContext: ...

    def execute(self, context: RunContext) -> RunContext: ...

    def persist(self, context: RunContext) -> RunContext: ...

    def fail(self, context: RunContext) -> RunContext: ...


class WorkflowMachine(StateChart[RunContext]):
    """Shared, declarative workflow for evaluation/training lifecycles."""

    catch_errors_as_events = True

    initialized = State(initial=True)
    prepared = State()
    built = State()
    executed = State()
    persisted = State()
    completed = State(final=True)
    failed = State(final=True)

    advance = (
        initialized.to(prepared)
        | prepared.to(built)
        | built.to(executed)
        | executed.to(persisted)
        | persisted.to(completed)
    )
    error_execution = (
        initialized.to(failed)
        | prepared.to(failed)
        | built.to(failed)
        | executed.to(failed)
        | persisted.to(failed)
    )

    def __init__(self, context: RunContext, runtime: WorkflowRuntime):
        self.context = context
        self.runtime = runtime
        super().__init__()

    def on_enter_initialized(self) -> None:
        self.advance()

    def on_enter_prepared(self) -> None:
        self.context = self.runtime.prepare(self.context)
        self.advance()

    def on_enter_built(self) -> None:
        self.context = self.runtime.build(self.context)
        self.advance()

    def on_enter_executed(self) -> None:
        self.context = self.runtime.execute(self.context)
        self.advance()

    def on_enter_persisted(self) -> None:
        self.context = self.runtime.persist(self.context)
        self.advance()

    def on_enter_failed(self, event=None, source=None, target=None, error=None) -> None:
        if error is not None:
            self.context.error = str(error)
        self.context = self.runtime.fail(self.context)

    def run(self) -> RunContext:
        return self.context
