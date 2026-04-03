"""Centralized initialization for MLflow tracing and DSPy autologging."""

import threading

import mlflow


class TracingManager:
    """Process-wide tracing initializer with idempotent locking."""

    _initialized = False
    _lock = threading.Lock()

    @classmethod
    def initialize(cls, *, enable_dspy_tracing: bool = True):
        """Configure MLflow tracing once; optionally enable DSPy autolog traces."""
        with cls._lock:
            if cls._initialized:
                return

            # Global MLflow tracing (lightweight)
            mlflow.autolog(log_traces=True, log_models=False, silent=True)

            if enable_dspy_tracing:
                mlflow.dspy.autolog(
                    log_traces=True,
                    log_traces_from_compile=False,
                    log_traces_from_eval=True,
                    log_compiles=False,
                    log_evals=False,
                    silent=True,
                )
            cls._initialized = True
