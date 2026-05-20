"""Per-sample scoring for the MAS evaluator.

This module owns the "given one prediction, build one
:class:`_ExampleEvalRecord`" pipeline shared by the deterministic
backend's metric callback (called once per sample inside
``dspy.Evaluate``) and the public single-example debug entry point.

A :class:`Scorer` is constructed with a :class:`MetadataExtractor` and
exposes:

* :meth:`score_prediction` — turn an existing ``(inputs, prediction)``
  pair into a fully populated record (no program invocation).
* :meth:`evaluate_one` — run the MAS program on one example and score
  the result; what the debug entry point uses.
* :meth:`resolve_label_and_consensus` — extract ``(y_true, y_pred)``
  for the downstream Fairlearn fairness aggregation.

Scorer methods are pure: they do not stream events, log to MLflow, or
mutate global state.  Side effects belong to
:mod:`bias_mitigation.mas.evaluation.pipeline`.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar, cast

import dspy

from ..metrics import (
    amplification_rate,
    build_round_bias_attribution,
    build_round_metric_series,
    emergence_rate,
    propagation_rate,
    system_robustness,
)
from .models import _AgentBiasSummary, _RoundMetricRow, _SampleOutcomeRow
from .worker import _ExampleEvalRecord

if TYPE_CHECKING:
    from .adapters import PredictFnAdapter
    from .metadata import MetadataExtractor


def feedback_value(feedback: Any) -> float:
    """Coerce a scorer output to a numeric value.

    MLflow GenAI scorers may return a plain ``float`` or a
    ``Feedback`` object whose ``.value`` carries the score.  Callers
    can rely on this helper to get a uniform numeric back.

    Args:
        feedback: Output of an MLflow ``@scorer`` function.

    Returns:
        The numeric score as a ``float``.
    """
    if hasattr(feedback, 'value'):
        return float(feedback.value)
    return float(feedback)


class Scorer:
    """Build per-sample :class:`_ExampleEvalRecord` instances.

    Encapsulates the four MAS metric extractors (system robustness,
    emergence, amplification, propagation), the per-agent-turn row
    builder (delegated to :class:`MetadataExtractor`), and the gold-
    label / consensus extraction used by the Fairlearn fairness path.

    Attributes:
        metadata_extractor: The evaluator's metadata extractor; supplies
            the run-wide defaults plus the agent-turn row builder.
        METRIC_NAME_MAP: Canonical metric keys used everywhere the
            evaluator surfaces these scalars (stream events, stratified
            rows, sample outcomes).
    """

    METRIC_NAME_MAP: ClassVar[dict[str, str]] = {
        'system_robustness': 'MAS_System_Robustness',
        'emergence_rate': 'MAS_Emergence_Rate',
        'amplification_rate': 'MAS_Amplification_Rate',
        'propagation_rate': 'MAS_Propagation_Rate',
    }

    def __init__(self, metadata_extractor: MetadataExtractor) -> None:
        """Hold a reference to the shared metadata extractor."""
        self.metadata_extractor = metadata_extractor

    def score_prediction(
        self,
        *,
        inputs: dict[str, Any],
        prediction: Any,
        sample_id: str,
        example_index: int,
        extra_metadata: dict[str, Any] | None,
    ) -> _ExampleEvalRecord:
        """Build a per-sample evaluation record from a pre-computed prediction.

        Used by both the ``dspy.Evaluate`` metric callback (program
        already invoked once by the evaluator) and the debug entry
        point.  The expensive work — extracting metrics, building
        per-agent-turn rows, computing per-round bias attribution —
        happens here exactly once per sample.

        Args:
            inputs: Example input dict (``example.toDict()``).
            prediction: Output of the MAS program for ``inputs``.
            sample_id: Stable per-sample identifier.
            example_index: Global example index (post-offset).
            extra_metadata: Optional per-call metadata overrides.

        Returns:
            A fully populated :class:`_ExampleEvalRecord` ready for
            stream emission and stratified aggregation.
        """
        metadata = self.metadata_extractor.extract(inputs, extra_metadata)

        metric_extractors = {
            self.METRIC_NAME_MAP['system_robustness']: system_robustness,
            self.METRIC_NAME_MAP['emergence_rate']: emergence_rate,
            self.METRIC_NAME_MAP['amplification_rate']: amplification_rate,
            self.METRIC_NAME_MAP['propagation_rate']: propagation_rate,
        }
        sample_metrics = {
            metric_name: feedback_value(scorer_fn(inputs=inputs, outputs=prediction))
            for metric_name, scorer_fn in metric_extractors.items()
        }
        sample_run_id = cast(str | None, getattr(prediction, 'sample_run_id', None))
        agent_turns = self.metadata_extractor.build_agent_turn_rows(
            inputs=inputs,
            prediction=prediction,
            sample_id=sample_id,
            example_index=example_index,
            metadata=metadata,
            sample_run_id=sample_run_id,
        )
        bias_summary = _AgentBiasSummary.from_turn_rows(agent_turns)

        sample_outcome = _SampleOutcomeRow(
            sample_id=sample_id,
            example_index=example_index,
            mlflow_run_id=metadata.run_id,
            dataset_name=metadata.dataset_name,
            dataset_source=metadata.dataset_source,
            stereoset_type=metadata.stereoset_type,
            category=metadata.category,
            protocol=metadata.protocol,
            llm_models=metadata.llm_models,
            model_names=metadata.model_names,
            intervention=metadata.intervention,
            num_agents=metadata.num_agents,
            rounds=metadata.rounds,
            split=metadata.split,
            seed=metadata.seed,
            question_polarity=str(inputs.get('question_polarity', 'unknown')),
            context_condition=str(inputs.get('context_condition', 'unknown')),
            label=inputs.get('label'),
            gold_answer_text=str(
                [inputs.get('ans0', ''), inputs.get('ans1', ''), inputs.get('ans2', '')][
                    int(inputs.get('label', 0))
                ]
            )
            if isinstance(inputs.get('label'), int)
            else 'unknown',
            system_robustness=sample_metrics[self.METRIC_NAME_MAP['system_robustness']],
            emergence_rate=sample_metrics[self.METRIC_NAME_MAP['emergence_rate']],
            amplification_rate=sample_metrics[self.METRIC_NAME_MAP['amplification_rate']],
            propagation_rate=sample_metrics[self.METRIC_NAME_MAP['propagation_rate']],
            turn_count=max(
                (len(turns) for turns in getattr(prediction, 'history', {}).values()), default=0
            ),
            processed_flag=True,
            failure_reason=None,
            sample_run_id=sample_run_id,
            agent_model_map=metadata.agent_model_map,
            first_biased_turn_by_agent=bias_summary.first_biased_turn_by_agent,
            final_is_biased_by_agent=bias_summary.final_is_biased_by_agent,
            final_answers=getattr(prediction, 'final_answers', {}),
        )
        round_bias_rows = build_round_bias_attribution([
            turn.model_dump(mode='python') for turn in agent_turns
        ])
        round_bias_by_turn = {int(row['turn_index']): row for row in round_bias_rows}
        round_metric_rows = [
            _RoundMetricRow.from_components(
                sample_id=sample_id,
                example_index=example_index,
                metadata=metadata,
                round_metrics=round_metrics,
                bias_row=round_bias_by_turn.get(int(round_metrics['turn_index'])),
            )
            for round_metrics in build_round_metric_series(inputs=inputs, outputs=prediction)
        ]

        y_true, y_pred = self.resolve_label_and_consensus(inputs, prediction)

        return _ExampleEvalRecord(
            metrics=sample_metrics,
            metadata=metadata,
            sample_outcome=sample_outcome,
            agent_turns=agent_turns,
            round_metric_rows=round_metric_rows,
            y_true=y_true,
            y_pred=y_pred,
        )

    def evaluate_one(
        self,
        *,
        predict_fn: PredictFnAdapter,
        example: dspy.Example,
        extra_metadata: dict[str, Any] | None,
        example_index: int,
    ) -> _ExampleEvalRecord:
        """Run the MAS program on a single example and score the result.

        Convenience for callers that have an example but not yet a
        prediction (the public single-example debug entry point on
        :class:`MASEvaluator` and ad-hoc analysis scripts).  The fast
        path inside the deterministic backend skips this method and
        calls :meth:`score_prediction` directly on the prediction
        produced by ``dspy.Evaluate``.

        Args:
            predict_fn: Adapter that invokes the underlying DSPy module.
            example: One ``dspy.Example`` from the devset.
            extra_metadata: Optional per-call metadata overrides.
            example_index: Global index of the example (post-offset).

        Returns:
            A fully populated :class:`_ExampleEvalRecord`.
        """
        inputs = example.toDict()
        sample_id = self.metadata_extractor.resolve_sample_id(inputs, example_index)
        prediction = predict_fn(**{**inputs, 'sample_id': sample_id})
        return self.score_prediction(
            inputs=inputs,
            prediction=prediction,
            sample_id=sample_id,
            example_index=example_index,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def resolve_label_and_consensus(
        inputs: dict[str, Any],
        prediction: Any,
    ) -> tuple[int | None, int | None]:
        """Extract gold label and consensus prediction indices for fairness eval.

        ``y_true`` is the integer label from the example.  ``y_pred`` is
        the index (in ``[ans0, ans1, ans2]``) of the most common final
        answer across agents; ties resolve to the lowest index.  Returns
        ``(None, None)`` if either piece cannot be parsed so Fairlearn
        aggregations can drop the row safely.

        Args:
            inputs: Raw inputs of a ``dspy.Example``.
            prediction: MAS program output (must expose ``final_answers``
                as ``dict[agent, answer]`` for the consensus path).

        Returns:
            ``(y_true, y_pred)`` where either component may be ``None``.
        """
        label = inputs.get('label')
        if not isinstance(label, int):
            return None, None
        options = [
            str(inputs.get('ans0', '')).strip(),
            str(inputs.get('ans1', '')).strip(),
            str(inputs.get('ans2', '')).strip(),
        ]
        final_answers = getattr(prediction, 'final_answers', None)
        if not isinstance(final_answers, dict) or not final_answers:
            return label, None
        indices = [
            options.index(str(answer).strip())
            for answer in final_answers.values()
            if isinstance(answer, str) and str(answer).strip() in options
        ]
        if not indices:
            return label, None
        counts = Counter(indices)
        top_count = max(counts.values())
        consensus = min(idx for idx, count in counts.items() if count == top_count)
        return label, consensus
