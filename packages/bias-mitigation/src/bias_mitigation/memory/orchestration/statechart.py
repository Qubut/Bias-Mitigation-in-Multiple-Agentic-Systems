"""Memory orchestration state machine for load-aware behavior."""

from statemachine import State, StateChart


class MemoryOrchestrationStateChart(StateChart[None]):
    """Transition memory mode based on success/failure pressure signals."""

    healthy = State(initial=True)
    degraded = State()
    shed = State()
    recovering = State()

    to_degraded = healthy.to(degraded) | recovering.to(degraded)
    to_shed = degraded.to(shed) | healthy.to(shed)
    to_recovering = shed.to(recovering) | degraded.to(recovering)
    to_healthy = recovering.to(healthy)

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
        if self._state_id() == self.healthy.id:
            return 'healthy'
        if self._state_id() == self.degraded.id:
            return 'degraded'
        if self._state_id() == self.shed.id:
            return 'shed'
        return 'recovering'

    def _state_id(self) -> str:
        return str(self.configuration[0].id)

    def note_success(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        if self._state_id() == self.shed.id:
            self.to_recovering()
            self.consecutive_successes = 1
            return
        if self._state_id() == self.recovering.id and (
            self.consecutive_successes >= self.recovery_success_threshold
        ):
            self.to_healthy()
            self.consecutive_successes = 0
            return
        if self._state_id() == self.degraded.id and (
            self.consecutive_successes >= self.recovery_success_threshold
        ):
            self.to_recovering()
            self.consecutive_successes = 1

    def note_failure(self) -> None:
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        if self.consecutive_failures < self.failure_trip_threshold:
            if self._state_id() == self.healthy.id:
                self.to_degraded()
            return

        if self._state_id() != self.shed.id:
            self.to_shed()
        self.consecutive_failures = 0

    def note_pressure(self) -> None:
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        if self._state_id() == self.healthy.id:
            self.to_degraded()
        elif self._state_id() == self.degraded.id:
            self.to_shed()
