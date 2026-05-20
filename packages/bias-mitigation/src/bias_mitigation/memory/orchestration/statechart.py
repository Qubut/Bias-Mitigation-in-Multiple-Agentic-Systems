"""Memory orchestration state machine for load-aware behavior.

The mode (``healthy`` → ``degraded`` → ``shed`` → ``recovering``) is
purely a function of the consecutive-success / consecutive-failure
counters that the orchestrator feeds in via :meth:`note_success`,
:meth:`note_failure`, and :meth:`note_pressure`.  Each public ``note_*``
method updates its counter, then triggers a single named event whose
:class:`State`-side ``cond=`` / ``unless=`` predicates let
python-statemachine pick the right edge — there is no imperative
``if state == ...`` ladder anywhere in this file.
"""

from statemachine import State, StateChart


class MemoryOrchestrationStateChart(StateChart[None]):
    """Transition memory mode based on success/failure pressure signals."""

    # Silent no-op when an event fires from a state with no matching edge,
    # e.g. ``note_success`` while already in ``healthy``.  This is what lets
    # us model the table as declarative transitions instead of guarded ``if``s.
    allow_event_without_transition = True

    healthy = State(initial=True, value='healthy')
    degraded = State(value='degraded')
    shed = State(value='shed')
    recovering = State(value='recovering')

    # Each ``note_*`` method below updates the counters and then fires the
    # corresponding event; the library evaluates the guards and picks the
    # right edge.  The trip / recovery thresholds are honoured via the
    # ``failure_tripped`` / ``recovery_complete`` predicates.
    failure_step = (
        healthy.to(degraded, unless='failure_tripped')
        | healthy.to(shed, cond='failure_tripped')
        | degraded.to(shed, cond='failure_tripped')
        | recovering.to(shed, cond='failure_tripped')
    )
    success_step = (
        shed.to(recovering)
        | recovering.to(healthy, cond='recovery_complete')
        | degraded.to(recovering, cond='recovery_complete')
    )
    pressure_step = healthy.to(degraded) | degraded.to(shed)

    def __init__(
        self,
        *,
        failure_trip_threshold: int,
        recovery_success_threshold: int,
    ):
        self.failure_trip_threshold = max(1, failure_trip_threshold)
        self.recovery_success_threshold = max(1, recovery_success_threshold)
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        super().__init__()

    @property
    def mode(self) -> str:
        return str(self.configuration[0].id)

    def failure_tripped(self) -> bool:
        return self.consecutive_failures >= self.failure_trip_threshold

    def recovery_complete(self) -> bool:
        return self.consecutive_successes >= self.recovery_success_threshold

    def note_failure(self) -> None:
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.failure_step()

    def note_success(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.success_step()

    def note_pressure(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.pressure_step()

    # State-entry hooks keep the post-transition counter resets next to the
    # state they belong to, rather than scattered through the ``note_*`` paths.
    def on_enter_recovering(self) -> None:
        self.consecutive_successes = 1

    def on_enter_healthy(self) -> None:
        self.consecutive_successes = 0

    def on_enter_shed(self) -> None:
        self.consecutive_failures = 0
