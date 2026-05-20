"""Typed configuration schema for the Bias-Mitigation MAS experiments.

This module is the single declarative source of truth for every knob the
research codebase exposes: dataset locations, per-agent LLM endpoints,
evaluator concurrency, Mem0 orchestration, GEPA optimization budgets, and
streaming/analysis output layout.  All experiment scripts (``train.py``,
``evaluate.py``, dataset downloaders) hydrate one of the Pydantic models
defined here from YAML files (optionally layered via OmegaConf overrides
and CLI dicts) and pass the validated object through the rest of the
pipeline.

Two top-level Pydantic ``BaseSettings`` models live here:

* :class:`AppConfig` — used by the dataset-download scripts; reads
  ``DOWNLOAD_*`` environment variables in addition to YAML.
* :class:`MASConfig` — the runtime configuration consumed by the MAS
  training / evaluation workflows; reads ``MAS_*`` environment variables.

Keeping every tunable in one validated schema means that changing an
experiment is a YAML edit rather than a code edit, which is essential for
reproducibility across the cluster.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from returns.result import safe

from bias_mitigation.data.models.memory_config import Mem0Config


class InterventionType(StrEnum):
    """Bias-mitigation interventions evaluated by this project.

    Each member names one experimental condition under study (memory on/off,
    prompt optimization on/off).  The string values are what appear in YAML
    config files and MLflow tags, so they form a stable public contract.

    Attributes:
        BASELINE: Vanilla cooperative debate with no memory and no prompt
            optimization — the control condition.
        baseline_opt: Baseline MAS whose prompts have been pre-optimized
            with GEPA; isolates the effect of prompt search from memory.
        MEM0G: Baseline MAS augmented with Mem0 vector-memory recall and
            store between rounds; isolates the effect of memory.
        MEM0G_GEPA: Full treatment combining Mem0 memory and GEPA-optimized
            prompts.
    """

    BASELINE = 'baseline'
    baseline_opt = 'baseline_opt'
    MEM0G = 'mem0g'
    MEM0G_GEPA = 'mem0g_gepa'


class BBQConfig(BaseModel):
    """Download configuration for the BBQ (Bias Benchmark for QA) dataset.

    BBQ is one of the two primary fairness benchmarks evaluated in this
    project.  Files are fetched per social-bias category from a single
    base URL.

    Attributes:
        base_url: Root URL where the per-category JSONL files live.  Must
            end with a trailing slash so categories can be appended
            verbatim; this invariant is enforced by ``validate_base_url``.
        categories: Subset of BBQ social-bias categories to download
            (e.g. ``"Age"``, ``"Gender_identity"``, ``"Race_ethnicity"``).
        dir_name: Local subdirectory under the datasets root in which the
            downloaded files are placed.
    """

    base_url: AnyUrl
    categories: list[str]
    dir_name: str = 'bbq'

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: AnyUrl) -> AnyUrl:
        """Enforce a trailing slash on ``base_url`` so URL joins are safe.

        Raises:
            ValueError: If ``v`` does not end with ``'/'``.
        """
        if not str(v).endswith('/'):
            raise ValueError("BBQ base_url must end with '/'")
        return v


class StereoSetConfig(BaseModel):
    """Download configuration for the StereoSet stereotype benchmark.

    Unlike BBQ, StereoSet ships as a small fixed set of named files rather
    than per-category endpoints, so the schema is a flat ``name -> url``
    mapping.

    Attributes:
        files: Mapping of logical file name (used as the local filename)
            to its source URL.
        dir_name: Local subdirectory under the datasets root in which the
            downloaded files are placed.
    """

    files: dict[str, AnyUrl]
    dir_name: str = 'stereoset'


class AppConfig(BaseSettings):
    """Top-level config for the dataset-download CLI.

    Loaded from a YAML file via :meth:`from_yaml` and optionally overridden
    by ``DOWNLOAD_*`` environment variables (e.g.
    ``DOWNLOAD_BBQ__BASE_URL=...``).  Extra keys are rejected to catch
    typos early.

    Attributes:
        bbq: BBQ-specific download settings.
        stereoset: StereoSet-specific download settings.
    """

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
        """Pin the settings-source precedence used by Pydantic-Settings.

        Order (highest priority first): explicit kwargs passed to ``__init__``,
        process environment variables, ``.env`` file, then file secrets.
        Returning the tuple explicitly makes the precedence part of the
        public contract instead of relying on Pydantic defaults.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    @safe(exceptions=(FileNotFoundError, yaml.YAMLError, ValueError))
    def from_yaml(cls, path: Path) -> 'AppConfig':
        """Load and validate an :class:`AppConfig` from a YAML file.

        Wrapped with :func:`returns.result.safe` so any of the expected I/O
        or parse errors are returned inside a ``Result`` rather than
        propagated, keeping the download-script call sites railway-oriented.

        Args:
            path: Filesystem path to the YAML config.

        Returns:
            A ``Result[AppConfig, Exception]`` containing the parsed config
            on success or the captured exception on failure.
        """
        with path.open(encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


class AgentModelConfig(BaseModel):
    """LLM endpoint description for one debating agent in the MAS.

    Each agent in a debate gets its own independent ``AgentModelConfig``,
    which keeps it possible to study heterogeneous-model debates (e.g.
    Llama-3 arguing with Qwen-2.5) and to fail-isolate one model's
    outage without disabling the run.

    Attributes:
        name: Provider-qualified model identifier passed to LiteLLM/DSPy
            (e.g. ``"openai/gpt-4o-mini"``, ``"openrouter/deepseek-r1"``).
        api_key: Secret token used to authenticate to ``api_base``.  Wrapped
            in :class:`SecretStr` so it never leaks into reprs or MLflow.
        api_base: Base URL of the OpenAI-compatible endpoint serving this
            model (vLLM, OpenRouter, OpenAI, etc.).
        temperature: Sampling temperature.  Defaults to ``0.0`` for the
            deterministic-evaluation regime; raise to introduce stochastic
            disagreement during debate.
        max_tokens: Per-call generation cap.
        context_window_tokens: Advertised context window for this model,
            used by the runtime when packing transcripts and tool outputs.
        agent_name: Logical role label this agent plays in the debate
            (e.g. ``"agent_a"``, ``"agent_b"``); also used as a key inside
            MAS state.
    """

    name: str
    api_key: SecretStr
    api_base: str
    temperature: float = 0.0
    max_tokens: int = 2048
    context_window_tokens: int = 16384
    agent_name: str


class EvaluatorConcurrencyConfig(BaseModel):
    """Bundle of concurrency and retry knobs for the evaluator.

    The evaluator runs each example through the MAS exactly once with
    bounded parallelism so that scoring is reproducible across re-runs.
    This config groups every limit that governs that loop — thread
    fan-out, per-endpoint inflight cap, timeouts, and retry backoff —
    so YAML files can override the whole policy in one block.

    Attributes:
        max_evaluation_threads: Worker count forwarded to
            ``dspy.Evaluate`` as ``num_threads``.
        max_llm_inflight_per_endpoint: Concurrent in-flight LLM calls
            allowed per unique ``(api_base, model)`` pair; protects shared
            vLLM/OpenRouter endpoints from thread storms.
        llm_timeout_seconds: Per-attempt network timeout applied to each
            LLM call.
        llm_retry_attempts: Maximum attempts per LLM call before giving up
            (initial attempt + retries).
        llm_retry_backoff_min_seconds: Lower bound of the exponential
            backoff delay between retries.
        llm_retry_backoff_max_seconds: Upper bound of the exponential
            backoff delay between retries.
    """

    max_evaluation_threads: int = Field(
        default=8,
        ge=1,
        description='Worker count forwarded to dspy.Evaluate (num_threads).',
    )
    max_llm_inflight_per_endpoint: int = Field(
        default=3,
        ge=1,
        description='Maximum concurrent LLM calls per endpoint (api_base + model).',
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description='Per-attempt network timeout for LLM calls.',
    )
    llm_retry_attempts: int = Field(
        default=3,
        ge=1,
        description='Maximum retry attempts per LLM call.',
    )
    llm_retry_backoff_min_seconds: float = Field(
        default=1.0,
        ge=0,
        description='Minimum exponential backoff delay between LLM retries.',
    )
    llm_retry_backoff_max_seconds: float = Field(
        default=4.0,
        ge=0,
        description='Maximum exponential backoff delay between LLM retries.',
    )


class MemoryOrchestrationConfig(BaseModel):
    """Worker-queue tuning for runtime Mem0 access by memory-based MAS.

    Mem0 recalls and stores are dispatched through a bounded worker pool
    with circuit-breaker thresholds so a slow or flaky memory backend
    cannot stall the debate loop.  Defaults are tuned for a small
    research-scale deployment.

    Attributes:
        worker_threads: Number of worker threads draining the Mem0 task
            queue.
        max_pending_store_tasks: Backpressure ceiling on the asynchronous
            store queue; additional stores beyond this are dropped or
            blocked by the orchestrator (see ``store_async``).
        recall_timeout_ms: Maximum wall-clock time the runtime waits on a
            Mem0 recall before falling back to memory-less behavior.
        store_timeout_ms: Maximum wait time for a Mem0 store operation
            when running synchronously.
        store_async: If ``True``, stores are dispatched fire-and-forget
            through the worker pool; if ``False``, the debate loop blocks
            until each store completes.
        failure_trip_threshold: Consecutive Mem0 failures after which the
            circuit breaker opens and recalls/stores short-circuit.
        recovery_success_threshold: Consecutive successes required after a
            trip before the breaker re-closes fully.
    """

    worker_threads: int = Field(default=4, ge=1)
    max_pending_store_tasks: int = Field(default=128, ge=1)
    recall_timeout_ms: int = Field(default=6000, ge=100)
    store_timeout_ms: int = Field(default=2500, ge=100)
    store_async: bool = True
    failure_trip_threshold: int = Field(default=8, ge=1)
    recovery_success_threshold: int = Field(default=6, ge=1)


class GepaConfig(BaseModel):
    """GEPA prompt-optimization controls (DSPy 3.x).

    Mirrors ``dspy.teleprompt.GEPA`` constructor arguments and is stored
    on :class:`MASConfig` so that the training workflow (which actually
    runs the optimizer) and the evaluation workflow (which only needs the
    save path to reload the optimized program) share a single declarative
    source.  Budgets are deliberately overlapping: GEPA accepts either a
    coarse profile (``auto``) or fine-grained call/eval ceilings, and at
    least one must be set.

    Attributes:
        auto: Budget preset — one of ``"light"``, ``"medium"``,
            ``"heavy"``, or ``None`` to fall back to the explicit budgets
            below.
        max_full_evals: Hard cap on full validation-set evaluations during
            optimization.  Used when ``auto`` is ``None``.
        max_metric_calls: Alternative budget expressed as total metric
            invocations across rollouts.
        num_threads: Parallelism for the inner GEPA evaluator.
        track_stats: If ``True``, GEPA records per-iteration statistics
            that the workflow persists to ``gepa/stats.json`` in MLflow.
        use_merge: Enable GEPA's program-merge proposer (off by default
            because it can blow up small search budgets).
        seed: PRNG seed for reproducible optimization runs.
        failure_score: Score assigned to a failed example so a single
            crash does not poison the Pareto front.
        reflection_lm_model: Optional override for the reflection LM that
            proposes new instructions; typically a stronger model than the
            task LM (e.g. GPT-4 reflecting over Llama-3 task rollouts).
        save_path: Where the optimized DSPy program is written via
            ``program.save``.  The evaluation workflow reads this path
            back when loading a previously-trained ``baseline_opt`` or
            ``mem0g_gepa`` checkpoint.
        valset_size: Stratified dev examples GEPA uses *during*
            optimization to track Pareto frontier.  Distinct from
            :attr:`validation_subset`, which is the post-GEPA hold-out
            check.  ``0`` disables and forces GEPA to score on the
            trainset (overfitting risk).
        validation_subset: Stratified dev examples used to validate the
            optimized program after GEPA finishes.  ``0`` skips the
            post-optimization validation phase entirely.
    """

    auto: str | None = Field(
        default='light',
        description="Budget profile: 'light' | 'medium' | 'heavy' | None (use max_full_evals/max_metric_calls).",
    )
    max_full_evals: int | None = Field(
        default=None,
        ge=1,
        description='Explicit budget when auto=None.',
    )
    max_metric_calls: int | None = Field(
        default=None,
        ge=1,
        description='Alternative explicit budget by metric-call count.',
    )

    @model_validator(mode='after')
    def _require_budget(self) -> 'GepaConfig':
        """Reject configs where no GEPA budget at all is set.

        Without at least one of ``auto``, ``max_full_evals``, or
        ``max_metric_calls``, GEPA finishes immediately with zero rollouts
        and returns the unmodified program.  That silent no-op is by far
        the most common cause of "my training run did nothing" tickets, so
        we fail loudly at config-load time instead.

        Raises:
            ValueError: If all three budget knobs are ``None``.
        """
        if self.auto is None and self.max_full_evals is None and self.max_metric_calls is None:
            raise ValueError(
                'GepaConfig: at least one of auto, max_full_evals, or max_metric_calls must be set. '
                'GEPA with no budget completes immediately with 0 rollouts.'
            )
        return self

    @computed_field
    @property
    def budget_kwargs(self) -> dict[str, Any]:
        """The exactly-one GEPA budget kwarg slice, in priority order.

        GEPA accepts exactly one of ``auto``, ``max_full_evals``, or
        ``max_metric_calls`` to bound its search; ``auto`` wins when
        present (matching upstream behaviour).  Exposing this as a
        computed field keeps the priority rule next to the validator that
        guarantees at least one knob is set.
        """
        for key in ('auto', 'max_full_evals', 'max_metric_calls'):
            value = getattr(self, key)
            if value is not None:
                return {key: value}
        return {}

    num_threads: int = Field(default=8, ge=1)
    track_stats: bool = True
    use_merge: bool = False
    seed: int = Field(default=42)
    failure_score: float = Field(default=0.0)
    reflection_lm_model: str | None = Field(
        default=None,
        description='Optional separate (stronger) LM model name for the reflection step.',
    )
    reflection_temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description=(
            'Sampling temperature for the GEPA reflection LM.  Reflection '
            'proposers benefit from diversity (non-zero) rather than the '
            "agents' task-time temperature; 1.0 is the upstream-recommended default."
        ),
    )
    reflection_fallback_max_tokens: int = Field(
        default=4096,
        ge=1,
        description=(
            'max_tokens for the reflection LM when ``reflection_lm_model`` is '
            "*not* mirrored from an ``agent_models`` entry (i.e. it's a hosted "
            'model resolved by LiteLLM via environment credentials).'
        ),
    )
    reflection_num_retries: int = Field(
        default=3,
        ge=0,
        description='LiteLLM retry budget for transient reflection-LM failures.',
    )
    save_path: str = Field(
        default='evaluation/checkpoints/gepa_optimized.json',
        description='Where to write the optimized DSPy program (program.save format).',
    )
    valset_size: int = Field(
        default=200,
        ge=0,
        description=(
            'Stratified dev examples used by GEPA *during* optimization (Pareto '
            'tracking).  Distinct from `validation_subset` (post-GEPA hold-out '
            'eval).  GEPA recommends the smallest valset that still matches the '
            'task distribution; 0 disables and falls back to trainset (overfits).'
        ),
    )
    validation_subset: int = Field(
        default=500,
        ge=0,
        description='Stratified dev examples used to validate the optimized program (0 disables).',
    )


class MASConfig(BaseSettings):
    """Runtime configuration for the MAS training and evaluation entry points.

    This is the master config consumed by ``train.py`` and ``evaluate.py``.
    It is loaded from YAML (via :meth:`load` or :meth:`load_merged`) and
    optionally overridden by ``MAS_*`` environment variables.  Extra keys
    are rejected so silent typos cannot ship to the cluster.

    A few groups of attributes work together:

    * Database selection: :attr:`db_url`.
    * MAS topology: :attr:`num_agents`, :attr:`rounds`, :attr:`protocol`,
      :attr:`malicious`, :attr:`agent_models`.
    * Intervention selection: :attr:`intervention`, :attr:`memory_config`.
    * Evaluator concurrency: :attr:`evaluator_*` fields plus the nested
      :attr:`evaluator_concurrency` block.
    * Streaming / analysis outputs: :attr:`analysis_*` and :attr:`stream_*`
      fields.
    * Memory orchestration: :attr:`memory_orchestration`.
    * Prompt optimization: :attr:`gepa`.

    Attributes:
        db_url: SQLAlchemy URL of the local dataset/results database; the
            default sqlite path is fine for single-node runs.
        num_agents: Number of debating agents instantiated per MAS
            program.  Must match the length of :attr:`agent_models`.
        rounds: Number of debate rounds per example.
        protocol: Debate protocol name (e.g. ``"cooperative"``).
        malicious: If ``True``, one agent is configured to argue
            adversarially — used by the robustness sub-study.
        evaluator_max_errors: Worker-error budget passed to
            ``dspy.Evaluate``; ``None`` means no early abort.
        evaluator_disable_progress_bar: Silences the ``dspy.Evaluate``
            progress bar; helpful for non-interactive job logs.
        evaluator_concurrency: Bundled concurrency / retry / timeout knobs
            (see :class:`EvaluatorConcurrencyConfig`).
        gepa: GEPA prompt-optimizer settings (see :class:`GepaConfig`).
        analysis_artifact_root: MLflow artifact prefix under which the
            analysis exporter writes its frozen, versioned outputs.
        analysis_local_root: Local directory the live streamer writes to
            while a run is in progress (mirrored to MLflow at the end).
        stream_flush_every_events: Flush local stream files every N
            emitted events; trade durability for throughput.
        stream_fsync: If ``True``, ``fsync`` each stream flush — costly
            but safe across machine crashes.
        stream_live_csv: If ``True``, mirror streamed rows into per-run
            CSV files for ad-hoc analysis.
        stream_max_buffered_events: Buffer cap before the backpressure
            policy kicks in.
        stream_drop_events_on_backpressure: When the buffer is full,
            ``True`` drops events (lossy but non-blocking); ``False``
            blocks emitters until space is available.
        analysis_live_dir_template: Format string controlling the
            human-readable directory names produced by the live streamer.
        analysis_live_slug_max_length: Per-token slug-length cap inside
            the directory template.
        analysis_live_write_manifest: If ``True``, drop a JSON manifest
            describing the run inside each live directory.
        analysis_live_index_filename: Root-level JSONL index that maps
            live directory names back to their run metadata.
        reset_memory_on_genesis: If ``True``, the state machine wipes
            session-scoped Mem0 entries at the start of each example so
            no cross-example leakage occurs.
        agent_models: Per-agent LLM endpoints (see
            :class:`AgentModelConfig`).  Length should equal
            :attr:`num_agents`.
        intervention: One of the :class:`InterventionType` values; selects
            the strategy used to wire memory and optimization.
        memory_config: Mem0 backend configuration.  Required when the
            intervention is memory-based (validated below).
        memory_orchestration: Worker / circuit-breaker knobs for the Mem0
            runtime queue (see :class:`MemoryOrchestrationConfig`).
    """

    model_config = SettingsConfigDict(extra='forbid', env_prefix='MAS_')
    db_url: str = 'sqlite+aiosqlite:///./datasets.db'
    num_agents: int = 2
    rounds: int = 4
    protocol: str = 'cooperative'
    malicious: bool = False
    evaluator_max_errors: int | None = Field(
        default=None,
        ge=1,
        description='Maximum failures tolerated by dspy.Evaluate before aborting the run.',
    )
    evaluator_disable_progress_bar: bool = Field(
        default=False,
        description='Disable the dspy.Evaluate progress bar.',
    )
    evaluator_concurrency: EvaluatorConcurrencyConfig = Field(
        default_factory=EvaluatorConcurrencyConfig,
        description='Grouped concurrency controls for the evaluator.',
    )
    gepa: GepaConfig = Field(
        default_factory=GepaConfig,
        description='GEPA optimizer controls used by the training workflow.',
    )
    analysis_artifact_root: str = Field(
        default='evaluation/analysis/v1',
        description='MLflow artifact root for analysis exports.',
    )
    analysis_local_root: str = Field(
        default='evaluation/analysis/live',
        description='Local directory root for live streaming analysis outputs.',
    )
    stream_flush_every_events: int = Field(
        default=1,
        ge=1,
        description='Flush local live stream files after this many emitted events.',
    )
    stream_fsync: bool = Field(
        default=False,
        description='If true, fsync local stream files for stronger durability.',
    )
    stream_live_csv: bool = Field(
        default=True,
        description='If true, write live CSV mirrors for stream rows.',
    )
    stream_max_buffered_events: int = Field(
        default=2048,
        ge=1,
        description='Maximum buffered stream events before applying backpressure policy.',
    )
    stream_drop_events_on_backpressure: bool = Field(
        default=False,
        description='If true, drop stream events when buffer is full instead of blocking emitters.',
    )
    analysis_live_dir_template: str = Field(
        default='{started_at}_{run_name}_{intervention}_{run_id_short}',
        description='Template used to build readable live analysis directory names.',
    )
    analysis_live_slug_max_length: int = Field(
        default=48,
        ge=8,
        description='Maximum slug length for each token used in live directory names.',
    )
    analysis_live_write_manifest: bool = Field(
        default=True,
        description='If true, write run manifest metadata inside each live analysis directory.',
    )
    analysis_live_index_filename: str = Field(
        default='runs_index.jsonl',
        description='Root-level JSONL index file mapping live directories to run metadata.',
    )
    reset_memory_on_genesis: bool = Field(
        default=False,
        description='If true, state machine clears session-scoped memory at genesis start.',
    )
    agent_models: list[AgentModelConfig] = Field(
        description='List of model names, where each agent gets its own independent LLM.'
    )
    intervention: str = Field(
        default='baseline',
        description="Must be one of: 'baseline', 'baseline_opt', 'mem0g', 'mem0g_gepa'",
    )
    memory_config: Mem0Config | None = Field(
        default=None,
        description='Configuration dictionary for Mem0 if intervention includes memory.',
    )
    memory_orchestration: MemoryOrchestrationConfig = Field(
        default_factory=MemoryOrchestrationConfig,
        description='State-machine orchestration and worker limits for Mem0 operations.',
    )

    @classmethod
    @safe(exceptions=(FileNotFoundError, yaml.YAMLError, ValueError, ValidationError))
    def load(cls, yaml_path: str = 'config.yaml') -> 'MASConfig':
        """Load a single YAML file (with OmegaConf interpolation) into a config.

        OmegaConf is used instead of plain ``yaml.safe_load`` so that
        ``${var}``-style references inside the YAML resolve before
        validation.  The result is wrapped in a :func:`returns.result.safe`
        ``Result`` so callers can chain failure handling.

        Args:
            yaml_path: Path to the YAML file.  Defaults to ``"config.yaml"``
            in the current working directory.

        Returns:
            ``Result[MASConfig, Exception]`` containing the validated config
            or the captured exception.
        """
        cfg = OmegaConf.load(yaml_path)
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    @safe(exceptions=(FileNotFoundError, yaml.YAMLError, ValueError, ValidationError))
    def load_merged(
        cls,
        base_path: str,
        *override_paths: str,
        cli_overrides: dict[str, Any] | None = None,
    ) -> 'MASConfig':
        """Layer multiple YAMLs and a CLI dict into one config, last write wins.

        This is the primary loader used by ``train.py`` and ``evaluate.py``
        to compose a base config (``mas_config.yaml``) with an
        intervention-specific override file and the per-invocation CLI
        flags, in that order.  Inside each layer OmegaConf merge semantics
        apply (deep-merge dicts, replace lists).

        Args:
            base_path: Path to the base YAML.
            *override_paths: Additional YAML files merged in order; later
                files override earlier ones.
            cli_overrides: Optional flat dict of last-mile overrides
                (typically built from ``argparse`` Namespace).

        Returns:
            ``Result[MASConfig, Exception]`` containing the validated,
            fully merged config.
        """
        base = OmegaConf.load(base_path)

        for path in override_paths:
            override = OmegaConf.load(path)
            base = OmegaConf.merge(base, override)

        if cli_overrides:
            cli_cfg = OmegaConf.create(cli_overrides)
            base = OmegaConf.merge(base, cli_cfg)

        return cls.model_validate(OmegaConf.to_container(base, resolve=True))

    @field_validator('memory_config')
    @classmethod
    def validate_memory_for_intervention(cls, v, info):
        """Reject memory-based interventions that omit a Mem0 configuration.

        Catches the common mistake of selecting ``mem0g`` or ``mem0g_gepa``
        without supplying the ``memory_config`` block, which would
        otherwise blow up much later at runtime when the strategy tries to
        wire up Mem0.

        Raises:
            ValueError: If the intervention requires memory but
                ``memory_config`` is unset.
        """
        intervention = info.data.get('intervention')
        if intervention in {InterventionType.MEM0G, InterventionType.MEM0G_GEPA} and not v:
            raise ValueError('memory_config required for Mem0g interventions')
        return v
