"""GEPA-based prompt optimization for the multi-agent bias-mitigation MAS."""

from __future__ import annotations

from typing import Any, Final

import dspy
import mlflow
from dspy.teleprompt import GEPA
from loguru import logger

from bias_mitigation.data.models.config import GepaConfig, MASConfig
from bias_mitigation.mas.metrics import make_paper_bias_metrics_gepa

#: Scalar engine metrics harvested off the GEPA optimizer instance.
_STATS_FROM_OPTIMIZER: Final[tuple[str, ...]] = (
    'best_score',
    'best_program_score',
    'num_metric_calls',
    'num_full_evals',
    'num_iterations',
    'pareto_front_size',
)
#: Structured artefacts attached to the optimized program (track_stats=True).
_STATS_FROM_PROGRAM: Final[tuple[str, ...]] = ('detailed_results', 'gepa_stats')


def _build_reflection_lm(mas_config: MASConfig) -> dspy.LM:
    """Build the GEPA reflection LM, mirroring an agent endpoint when named."""
    gepa_cfg = mas_config.gepa
    name = gepa_cfg.reflection_lm_model
    if not name:
        raise ValueError(
            'GEPA requires gepa.reflection_lm_model — set it to one of '
            'agent_models[*].name or any LiteLLM-resolvable model id.',
        )
    agent = next((a for a in mas_config.agent_models if a.name == name), None)
    if agent is None:
        return dspy.LM(
            model=name,
            temperature=gepa_cfg.reflection_temperature,
            max_tokens=gepa_cfg.reflection_fallback_max_tokens,
        )
    return dspy.LM(
        model=agent.name,
        api_key=agent.api_key.get_secret_value(),
        api_base=agent.api_base,
        cache=False,
        model_type='chat',
        temperature=gepa_cfg.reflection_temperature,
        max_tokens=agent.max_tokens,
        num_retries=gepa_cfg.reflection_num_retries,
        timeout=mas_config.evaluator_concurrency.llm_timeout_seconds,
    )


def _harvest_stats(optimizer: GEPA, optimized: dspy.Module) -> dict[str, Any]:
    """Merge optimizer scalars and program-side artefacts into one stats dict."""
    from_optimizer = {
        attr: value
        for attr in _STATS_FROM_OPTIMIZER
        if (value := getattr(optimizer, attr, None)) is not None
    }
    from_program = {
        attr: value
        for attr in _STATS_FROM_PROGRAM
        if isinstance(value := getattr(optimized, attr, None), (dict, list))
    }
    return {**from_optimizer, **from_program}


def gepa_optimize_mas(
    program: dspy.Module,
    trainset: list[dspy.Example],
    config: GepaConfig | None = None,
    mas_config: MASConfig | None = None,
    valset: list[dspy.Example] | None = None,
) -> tuple[dspy.Module, dict[str, Any]]:
    """Optimize a MAS program with GEPA against the paper's bias metric.

    Args:
        program: DSPy module to optimize (typically a :class:`MASProgram`).
            GEPA deep-copies this per candidate; see ``MASProgram.__deepcopy__``
            for the share-by-reference policy on runtime resources.
        trainset: Labeled examples used to score candidates.
        config: GEPA engine knobs.  Defaults to ``mas_config.gepa``.
        mas_config: Full MAS config.  Required to resolve the reflection
            LM's endpoint/credentials from ``agent_models``.
        valset: GEPA's Pareto-tracking validation set.  **Not** the
            post-GEPA hold-out — that lives in the workflow.  When omitted
            GEPA reuses ``trainset`` (the DSPy docs warn this overfits).

    Returns:
        ``(optimized_program, stats)``.  ``stats`` merges engine scalars
        (best score, metric calls, …) with the per-iteration trace.

    Raises:
        ValueError: if ``mas_config`` is missing or its
            ``gepa.reflection_lm_model`` is unset.
    """
    if mas_config is None:
        raise ValueError(
            'gepa_optimize_mas requires mas_config to resolve the reflection LM. '
            'Pass mas_config=... explicitly.',
        )
    cfg = config or mas_config.gepa

    if mlflow.active_run() is None:
        logger.warning(
            '[gepa]: no active MLflow run — engine per-iteration metrics will be '
            'dropped silently.  Wrap with mlflow.start_run() (the training workflow '
            'normally does this).',
        )

    optimizer = GEPA(
        metric=make_paper_bias_metrics_gepa(failure_score=cfg.failure_score),
        num_threads=cfg.num_threads,
        track_stats=cfg.track_stats,
        use_merge=cfg.use_merge,
        seed=cfg.seed,
        failure_score=cfg.failure_score,
        reflection_lm=_build_reflection_lm(mas_config),
        use_mlflow=True,
        **cfg.budget_kwargs,
    )
    compile_kwargs: dict[str, Any] = {'trainset': trainset}
    if valset:
        compile_kwargs['valset'] = valset
    optimized = optimizer.compile(program, **compile_kwargs)
    return optimized, _harvest_stats(optimizer, optimized)
