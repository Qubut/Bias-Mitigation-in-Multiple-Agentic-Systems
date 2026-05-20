"""Declarative training workflow that reuses evaluation stages and swaps ``execute``.

The MAS lifecycle is modelled as a five-stage Pipeline declared by the
``WorkflowRuntime`` Protocol in :mod:`bias_mitigation.workflows.statechart`:
``prepare`` -> ``build`` -> ``execute`` -> ``persist`` (with ``fail`` as the
error path).  The evaluation workflow already implements every stage; the
training workflow needs an *almost identical* shape — same dataset
hydration, same MAS construction, same MLflow persistence — but with GEPA
optimization in the middle instead of plain evaluation.

Rather than duplicate those four stages, this module applies the Strategy
pattern to the inner pipeline:

* :class:`TrainingWorkflowRuntimeImpl` reuses ``prepare`` / ``build`` /
  ``fail`` verbatim from :mod:`bias_mitigation.workflows.evaluation`.
* It overrides ``execute`` with :func:`_gepa_optimize`, which calls GEPA
  and (optionally) runs a hold-out validation pass.
* It overrides ``persist`` with :func:`_persist_training_outputs`, which
  delegates to the evaluation persister and then logs the optimized
  program plus GEPA stats as extra MLflow artefacts.

The result is a DRY training entry point: every lifecycle concern that
matters for evaluation (run context, MLflow tags, streaming, failure
handling) is automatically correct for training as well.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import dspy
import mlflow
from loguru import logger
from returns.iterables import Fold
from returns.maybe import Maybe, Nothing, Some

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.mas.optimization import gepa_optimize_mas

from .contracts import RunContext
from .evaluation import (
    _build_components,
    _execute_evaluation,
    _handle_failure,
    _persist_outputs,
    _prepare_context,
    _try_log,
)
from .statechart import WorkflowRuntime


def _gepa_optimize(context: RunContext) -> RunContext:
    """``execute`` stage of the training pipeline.

    Performs the type-narrowing of the run context's optional fields once,
    then dispatches to a short sequence of single-responsibility helpers
    (``_build_valset`` -> ``_run_gepa`` -> ``_save_optimized`` ->
    ``_validate_or_skip`` -> ``_attach_result_metadata``).  Each helper
    declares its real dependencies in its signature so we avoid the
    "every function asserts not-None on every field" anti-pattern.

    Args:
        context: The run context populated by ``prepare`` and ``build``.

    Returns:
        The same context, mutated in place with the optimized program,
        GEPA stats, and (optionally) hold-out evaluation results.

    Raises:
        RuntimeError: If required context fields are missing or if the
            train split is empty.
    """
    mas_config = _require(context.mas_config, 'MASConfig')
    _require(context.mas_program, 'mas_program')
    train_examples = context.train_examples
    if not train_examples:
        raise RuntimeError(
            'Empty trainset — verify dataset_dir and that load_and_track_splits '
            'returned a non-empty train split.'
        )

    logger.info(
        f'[train]: context sizes — '
        f'train={len(train_examples)}, '
        f'gepa_val={len(context.gepa_val_examples)}, '
        f'holdout={len(context.holdout_examples)}, '
        f'max_metric_calls={mas_config.gepa.max_metric_calls}, '
        f'auto={mas_config.gepa.auto!r}'
    )

    valset = _build_valset(context, mas_config)
    optimized = _run_gepa(context, mas_config, train_examples, valset)
    _save_optimized(context, mas_config, optimized)
    _validate_or_skip(context, mas_config, optimized, train_examples)
    _attach_result_metadata(context)
    return context


def _require(value: Any, name: str) -> Any:
    """Type-narrowing precondition that converts ``None`` into a hard error.

    Used at the top of :func:`_gepa_optimize` to convert the optional
    fields of :class:`RunContext` into non-optional locals exactly once,
    so downstream helpers can declare them as required in their signatures.

    Args:
        value: The optional value being checked.
        name: Human-readable field name used in the error message.

    Returns:
        ``value`` unchanged (with ``None`` excluded).

    Raises:
        RuntimeError: If ``value`` is ``None``.
    """
    if value is None:
        raise RuntimeError(f'Workflow context missing {name} at training step.')
    return value


def _build_valset(context: RunContext, mas_config: MASConfig) -> list[Any] | None:
    """Pick the validation set that GEPA uses for Pareto tracking.

    GEPA needs a stratified dev set distinct from the trainset to track
    candidate programs along the Pareto frontier.  The run context is
    expected to already carry that split (``gepa_val_examples``) which is
    guaranteed non-overlapping with ``holdout_examples`` used for post-GEPA
    validation.

    Returns:
        The pre-split GEPA validation list, or ``None`` if the split was
        empty (in which case GEPA falls back to the trainset).
    """
    if not context.gepa_val_examples:
        return None
    return context.gepa_val_examples


_THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


def _strip_one_think_tag(named: tuple[str, Any]) -> Maybe[str]:
    """Strip a ``<think>`` block from a single predictor's instructions.

    Operates on one ``(name, predictor)`` tuple as produced by
    ``program.named_predictors()`` so it can be ``map``-ed across the
    whole program inside :func:`_strip_think_tags`.

    Args:
        named: ``(predictor_name, predictor)`` pair from a DSPy program.

    Returns:
        ``Some(name)`` when the predictor's instructions actually changed
        (useful for debug logging and Fold accumulation), otherwise
        ``Nothing``.
    """
    name, predictor = named
    sig = predictor.signature
    cleaned = _THINK_PATTERN.sub('', sig.instructions).strip()
    if cleaned == sig.instructions:
        return Nothing
    predictor.signature = sig.with_instructions(cleaned)
    logger.debug(f'[train]: stripped <think> block from predictor "{name}"')
    return Some(name)


def _strip_think_tags(optimized: Any) -> None:
    """Remove ``<think>...</think>`` reasoning blocks from every predictor.

    GEPA uses an LLM to propose new signature instructions.  When the
    proposer is a reasoning model (DeepSeek-R1, QwQ, ...) the full
    chain-of-thought is appended to the instruction string inside
    ``<think>`` tags, which both inflates context for downstream calls and
    leaks proposer-internal reasoning into the saved program.  This pass
    iterates over every predictor and strips those blocks in place using
    a declarative ``Fold.collect_all`` over :func:`_strip_one_think_tag`.

    Args:
        optimized: The DSPy program returned by GEPA; mutated in place.
    """
    Fold.collect_all(
        map(_strip_one_think_tag, optimized.named_predictors()),
        Some(()),
    )


def _run_gepa(
    context: RunContext,
    mas_config: MASConfig,
    train_examples: list[Any],
    valset: list[Any] | None,
) -> Any:
    """Invoke GEPA to optimize the MAS prompts and update the run context.

    Before the call this helper re-asserts two pieces of DSPy/MLflow
    configuration that are easy to lose between processes:

    * ``log_traces_from_compile=True`` so every GEPA iteration is captured
      as an MLflow trace (otherwise only eval traces show up and debugging
      a bad optimization is much harder).
    * ``dspy.settings.configure(provide_traceback=True)`` so DSPy forwards
      full tracebacks on LLM failures instead of swallowing them.

    On return the helper writes the optimized program and stats
    (including the wall-clock duration) back onto ``context`` so later
    pipeline stages can persist them.

    Args:
        context: Active run context; mutated in place.
        mas_config: Validated runtime config providing the GEPA budget.
        train_examples: Trainset passed to GEPA.
        valset: Optional Pareto-tracking validation set; ``None`` makes
            GEPA fall back to scoring on the trainset.

    Returns:
        The optimized DSPy program (also stored on
        ``context.optimized_program``).
    """
    # Re-assert compile tracing: ``mlflow.dspy.autolog`` is idempotent and
    # last-write-wins, so this safely upgrades the eval-mode config installed
    # by the build stage to capture GEPA spans for this training pass.
    mlflow.dspy.autolog(
        log_traces=True,
        log_traces_from_compile=True,
        log_traces_from_eval=True,
        log_compiles=False,
        log_evals=False,
        silent=True,
    )
    dspy.settings.configure(provide_traceback=True)
    gepa_cfg = mas_config.gepa
    logger.info(
        f'[train]: starting GEPA optimization '
        f'(auto={gepa_cfg.auto}, num_threads={gepa_cfg.num_threads}, '
        f'trainset_size={len(train_examples):,}, '
        f'valset_size={len(valset) if valset else 0})'
    )
    started = time.monotonic()
    optimized, gepa_stats = gepa_optimize_mas(
        program=context.mas_program,
        trainset=train_examples,
        config=gepa_cfg,
        mas_config=mas_config,
        valset=valset,
    )
    elapsed = time.monotonic() - started
    logger.info(f'[train]: GEPA finished in {elapsed:.1f}s')
    _strip_think_tags(optimized)
    if gepa_stats:
        logger.info(f'[train]: GEPA stats: {gepa_stats}')
    context.optimized_program = optimized
    context.gepa_stats = {**gepa_stats, 'elapsed_seconds': elapsed}
    return optimized


def _save_optimized(context: RunContext, mas_config: MASConfig, optimized: Any) -> None:
    """Write the optimized DSPy program to disk on a best-effort basis.

    The destination is either the explicit path requested for this run
    (``context.request.optimized_program_path``) or the default from
    ``mas_config.gepa.save_path``.  Parent directories are created
    automatically.  Any save error is logged with full traceback but does
    not abort the pipeline — the optimized program is still in memory and
    the hold-out evaluation can still run.
    """
    save_path = context.request.optimized_program_path or Path(mas_config.gepa.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        optimized.save(str(save_path))
        context.optimized_program_path = save_path
        logger.success(f'[train]: optimized program saved → {save_path}')
    except Exception as exc:
        logger.exception(f'[train]: optimized program save failed: {exc}')


def _validate_or_skip(
    context: RunContext,
    mas_config: MASConfig,
    optimized: Any,
    train_examples: list[Any],
) -> None:
    """Run a leak-free hold-out evaluation of the optimized program, or skip.

    The hold-out set (``context.holdout_examples``) is guaranteed to be
    disjoint from both the GEPA trainset and the GEPA Pareto valset, so
    the metrics produced here are an honest estimate of generalization
    after prompt optimization.

    Validation is skipped when:

    * the caller passed ``--skip-validation`` (``request.skip_validation``);
      typically used for cheap smoke tests, or
    * the hold-out split is empty (small experimental setups).

    When skipped this helper still populates ``context.result`` with a
    minimal placeholder dict (``phase='gepa_train_only'``) so the persist
    stage has something coherent to log.  When run, it delegates to the
    evaluation helper :func:`_execute_evaluation` and tags the result with
    ``phase='gepa_validation'`` and the hold-out size.
    """
    should_validate = not context.request.skip_validation and bool(context.holdout_examples)
    if not should_validate:
        logger.info('[train]: validation skipped (per request or empty holdout set).')
        context.result = {
            'phase': 'gepa_train_only',
            'overall_metrics': {},
            'failure_count': 0,
            'processed_count': len(train_examples),
            'gepa_stats': context.gepa_stats,
        }
        return

    holdout = context.holdout_examples
    logger.info(f'[train]: validating optimized program on hold-out set (n={len(holdout)})')
    context.mas_program = optimized
    context.evaluated_examples = holdout
    _execute_evaluation(context)
    context.result.setdefault('phase', 'gepa_validation')
    context.result.setdefault('validation_subset_size', len(holdout))


def _attach_result_metadata(context: RunContext) -> None:
    """Copy GEPA artefacts into ``context.result`` for downstream logging.

    The persist stage (:func:`_persist_outputs` from
    :mod:`bias_mitigation.workflows.evaluation`) writes whatever is in
    ``context.result`` to MLflow; this helper ensures the GEPA stats dict
    and the path to the saved program are surfaced through that channel
    without disturbing values the evaluation step may already have set
    (hence the use of ``setdefault``).
    """
    context.result.setdefault('gepa_stats', context.gepa_stats)
    if context.optimized_program_path is not None:
        context.result.setdefault(
            'optimized_program_path',
            str(context.optimized_program_path),
        )


def _persist_training_outputs(context: RunContext) -> RunContext:
    """``persist`` stage for training: reuse eval persist, then add GEPA artefacts.

    The shared evaluation persister handles the bulk of the work (params,
    metrics, streamed analysis files).  This wrapper extends it with two
    training-specific MLflow logs:

    * the saved optimized program file, attached under the ``gepa/``
      artifact prefix;
    * the GEPA stats dict, serialised through :func:`_json_safe` and
      written as ``gepa/stats.json``.

    Both extensions use the Railway-Oriented :func:`_try_log` helper from
    :mod:`bias_mitigation.workflows.evaluation` so a single MLflow hiccup
    cannot abort the rest of persistence.  ``Maybe.from_optional`` is
    preferred over ``if x is not None`` to keep the style consistent with
    the rest of the workflow.
    """
    context = _persist_outputs(context)

    # Log the optimized program file as an artifact under the 'gepa/' prefix.
    Maybe.from_optional(context.optimized_program_path).map(
        lambda p: _try_log(
            'log_artifact(optimized_program)',
            lambda: mlflow.log_artifact(str(p), artifact_path='gepa'),
        )
    )

    # Log GEPA optimizer stats as a structured JSON dict.
    if context.gepa_stats:
        _try_log(
            'log_dict(gepa_stats)',
            lambda: mlflow.log_dict(
                {k: _json_safe(v) for k, v in context.gepa_stats.items()},
                'gepa/stats.json',
            ),
        )

    return context


def _json_safe(v: Any) -> Any:
    """Recursively coerce arbitrary GEPA stat values into JSON-serialisable form.

    GEPA's stats dict can contain numpy scalars, custom dataclasses, or
    other non-JSON types depending on the optimizer version.  This helper
    keeps primitives, recurses into containers, and falls back to ``str()``
    for anything exotic so ``mlflow.log_dict`` never fails on serialization.

    Args:
        v: Any value pulled out of the GEPA stats dict.

    Returns:
        A value composed only of ``str``/``int``/``float``/``bool``/``None``
        primitives, ``list``s, and ``dict``s.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return str(v)


class TrainingWorkflowRuntimeImpl:
    """``WorkflowRuntime`` implementation specialised for GEPA-based training.

    Implements the five-stage Pipeline contract by *delegating* the
    untouched stages (``prepare``, ``build``, ``fail``) to the evaluation
    workflow helpers and providing training-specific implementations only
    for ``execute`` (GEPA optimization + optional hold-out validation)
    and ``persist`` (eval persist + extra GEPA artefacts).  This is the
    Strategy pattern at the pipeline-stage level, which is what keeps
    train and evaluate genuinely DRY across the codebase.
    """

    def prepare(self, context: RunContext) -> RunContext:
        """Delegate to the evaluation ``prepare`` stage (dataset / MLflow setup)."""
        return _prepare_context(context)

    def build(self, context: RunContext) -> RunContext:
        """Delegate to the evaluation ``build`` stage (MAS construction)."""
        return _build_components(context)

    def execute(self, context: RunContext) -> RunContext:
        """Run GEPA optimization and (optionally) hold-out validation."""
        return _gepa_optimize(context)

    def persist(self, context: RunContext) -> RunContext:
        """Reuse evaluation persistence and log GEPA artefacts on top."""
        return _persist_training_outputs(context)

    def fail(self, context: RunContext) -> RunContext:
        """Delegate to the evaluation ``fail`` handler for consistent error handling."""
        return _handle_failure(context)


def build_training_workflow_runtime() -> WorkflowRuntime:
    """Build a fresh :class:`TrainingWorkflowRuntimeImpl` for the state machine.

    Mirrors ``build_evaluation_workflow_runtime`` so ``train.py`` and
    ``evaluate.py`` are symmetric: each script just calls its own builder
    and hands the result to the shared state-chart driver.
    """
    return TrainingWorkflowRuntimeImpl()
