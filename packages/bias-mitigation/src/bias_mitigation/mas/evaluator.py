"""Evaluation helpers for deterministic and GenAI MAS scoring backends."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from math import sqrt
from statistics import fmean, pstdev
from typing import Any, ClassVar

import dspy
import mlflow
from mlflow.genai import evaluate

from .metrics import (
    amplification_rate,
    emergence_rate,
    propagation_rate,
    system_robustness,
)
from .prediction_models import AgentHistory, AgentTurn, FinalAnswers, MASPredictionEnvelope


class EvaluatorBackend(StrEnum):
    """Available evaluator backend options."""

    DETERMINISTIC = 'deterministic'
    GENAI = 'genai'


@dataclass(slots=True)
class _ExampleEvalRecord:
    """Per-example evaluation values plus stratification metadata."""

    metrics: dict[str, float]
    metadata: dict[str, str]


@dataclass(slots=True)
class _StratifiedRow:
    """Typed stratified aggregate row."""

    dimensions: dict[str, str]
    support: int
    metrics: dict[str, float]
    ci95: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """Serialize row to plain mapping."""
        return {
            'dimensions': self.dimensions,
            'support': self.support,
            'metrics': self.metrics,
            'ci95': self.ci95,
        }


@dataclass(slots=True)
class _EvaluatorOutput:
    """Typed evaluator output contract."""

    backend: str
    system_robustness: float
    genai_evaluation: Any
    detailed_results: list[dict[str, Any]] | None
    config: Any
    genai_metrics: dict[str, Any]
    stratified_metrics: list[dict[str, Any]]
    uncertainty: dict[str, float]
    failure_count: int
    processed_count: int
    failed_examples: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Serialize evaluator output to plain mapping."""
        return {
            'backend': self.backend,
            'system_robustness': self.system_robustness,
            'genai_evaluation': self.genai_evaluation,
            'detailed_results': self.detailed_results,
            'config': self.config,
            'genai_metrics': self.genai_metrics,
            'stratified_metrics': self.stratified_metrics,
            'uncertainty': self.uncertainty,
            'failure_count': self.failure_count,
            'processed_count': self.processed_count,
            'failed_examples': self.failed_examples,
        }


class PredictFnAdapter:
    """Adapter that exposes a DSPy module as an MLflow ``predict_fn``."""

    def __init__(self, program: dspy.Module):
        self.program = program

    @staticmethod
    def _coerce_turn(turn: Any, agent_name: str) -> AgentTurn:
        """Convert one raw turn object into validated ``AgentTurn`` model."""
        if isinstance(turn, dspy.Prediction):
            payload = {
                'answer': getattr(turn, 'answer', None),
                'reasoning': getattr(turn, 'reasoning', None),
                'agent_name': getattr(turn, 'agent_name', agent_name),
            }
        elif isinstance(turn, dict):
            payload = {
                'answer': turn.get('answer'),
                'reasoning': turn.get('reasoning'),
                'agent_name': turn.get('agent_name', agent_name),
            }
        else:
            raise TypeError(f'Invalid turn type in history for {agent_name}: {type(turn)!r}')

        return AgentTurn.model_validate(payload)

    @classmethod
    def _normalize_prediction(cls, prediction: dspy.Prediction) -> dspy.Prediction:
        """Validate and normalize prediction payload via Pydantic schema contract."""
        history = getattr(prediction, 'history', None)
        final_answers = getattr(prediction, 'final_answers', None)

        if not isinstance(history, dict):
            raise TypeError(
                f'MASProgram prediction missing valid history mapping: {prediction!r}'
            )
        if not isinstance(final_answers, dict):
            raise TypeError(
                f'MASProgram prediction missing valid final_answers mapping: {prediction!r}'
            )

        typed_history = {
            agent_name: AgentHistory(
                turns=[cls._coerce_turn(turn, agent_name) for turn in turns]
            )
            for agent_name, turns in history.items()
        }
        envelope = MASPredictionEnvelope(
            history=typed_history,
            final_answers=FinalAnswers(values=final_answers),
        )

        if not envelope.history:
            raise ValueError('Prediction history is empty; cannot score MAS metrics.')

        for agent_name, agent_history in envelope.history.items():
            if not agent_history.turns:
                raise ValueError(f'Prediction history for {agent_name} has no turns.')

        history_agents = set(envelope.history.keys())
        final_answer_agents = set(envelope.final_answers.values.keys())
        if history_agents != final_answer_agents:
            raise ValueError(
                'Mismatch between history agent keys and final_answers keys: '
                f'history={sorted(history_agents)}, final_answers={sorted(final_answer_agents)}'
            )

        normalized_history = {
            agent_name: [
                dspy.Prediction(
                    answer=turn.answer,
                    reasoning=turn.reasoning,
                    agent_name=turn.agent_name or agent_name,
                )
                for turn in turns.turns
            ]
            for agent_name, turns in envelope.history.items()
        }

        return dspy.Prediction(
            history=normalized_history,
            final_answers=dict(envelope.final_answers.values),
        )

    @mlflow.trace(name='PredictFnAdapter_call', span_type=mlflow.entities.SpanType.AGENT)
    def __call__(self, **inputs: Any) -> dspy.Prediction:
        """Forward a normalized input mapping to ``program.forward``."""
        # Forward only the fields the MASProgram expects
        prediction = self.program(
            context=inputs['context'],
            question=inputs['question'],
            ans0=inputs['ans0'],
            ans1=inputs['ans1'],
            ans2=inputs['ans2'],
            category=inputs['category'],
            stereotyped_groups=inputs.get('stereotyped_groups'),
            **{
                k: v
                for k, v in inputs.items()
                if k
                not in {
                    'context',
                    'question',
                    'ans0',
                    'ans1',
                    'ans2',
                    'category',
                    'stereotyped_groups',
                }
            },
        )

        if prediction is None or not isinstance(prediction, dspy.Prediction):
            raise RuntimeError(f'MASProgram returned invalid prediction object: {prediction!r}')

        return self._normalize_prediction(prediction)


class MASEvaluator:
    """Run MAS evaluation with deterministic default and optional GenAI backend."""

    _METRIC_NAME_MAP: ClassVar[dict[str, str]] = {
        'system_robustness': 'MAS_System_Robustness',
        'emergence_rate': 'MAS_Emergence_Rate',
        'amplification_rate': 'MAS_Amplification_Rate',
        'propagation_rate': 'MAS_Propagation_Rate',
    }

    _STRATIFY_FIELDS: ClassVar[tuple[str, ...]] = (
        'dataset_name',
        'category',
        'protocol',
        'llm_models',
        'intervention',
        'num_agents',
        'rounds',
        'split',
        'seed',
    )

    _BBQ_CATEGORIES: ClassVar[set[str]] = {
        'Age',
        'Disability_status',
        'Gender_identity',
        'Nationality',
        'Physical_appearance',
        'Race_ethnicity',
        'Race_x_gender',
        'Race_x_SES',
        'Religion',
        'SES',
        'Sexual_orientation',
    }

    _STEREOSET_CATEGORIES: ClassVar[set[str]] = {
        'intersentence',
        'intrasentence',
        'dev',
    }

    def __init__(
        self,
        devset: list[dspy.Example],
        backend: EvaluatorBackend | str = EvaluatorBackend.DETERMINISTIC,
        run_metadata: dict[str, Any] | None = None,
    ):
        self.devset = devset
        self.backend = EvaluatorBackend(backend)
        self.run_metadata = run_metadata or {}

    @staticmethod
    def _extract_system_robustness(metrics: dict[str, Any]) -> float:
        """Extract system robustness metric from MLflow GenAI metric mapping."""
        direct_key = 'MAS_System_Robustness'
        if direct_key in metrics:
            return float(metrics[direct_key])

        for key, value in metrics.items():
            if 'MAS_System_Robustness' in key:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        return 0.0

    def _to_mlflow_eval_dataset(self, devset: list[dspy.Example]) -> list[dict[str, Any]]:
        """Convert DSPy examples into MLflow GenAI dataset records."""
        return [{'inputs': example.toDict()} for example in devset]

    @staticmethod
    def _feedback_value(feedback: Any) -> float:
        """Extract metric value from a Feedback-like scorer return."""
        if hasattr(feedback, 'value'):
            return float(feedback.value)
        return float(feedback)

    def _extract_metadata(
        self,
        inputs: dict[str, Any],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, str]:
        """Build stratification metadata for one evaluated sample."""
        merged: dict[str, Any] = {}
        merged.update(self.run_metadata)
        if extra_metadata:
            merged.update(extra_metadata)

        category = inputs.get('category') or merged.get('category') or 'unknown'

        dataset_name = (
            inputs.get('dataset_name')
            or inputs.get('dataset')
            or merged.get('dataset_name')
            or self._infer_dataset_name(str(category))
        )

        return {
            'dataset_name': str(dataset_name),
            'category': str(category),
            'protocol': str(merged.get('protocol', 'unknown')),
            'llm_models': str(merged.get('llm_models', 'unknown')),
            'intervention': str(merged.get('intervention', 'unknown')),
            'num_agents': str(merged.get('num_agents', 'unknown')),
            'rounds': str(merged.get('rounds', 'unknown')),
            'split': str(merged.get('split', 'unknown')),
            'seed': str(merged.get('seed', 'unknown')),
            'run_id': str(merged.get('run_id', 'unknown')),
        }

    @classmethod
    def _infer_dataset_name(cls, category: str) -> str:
        """Infer dataset family from category label when dataset_name is missing."""
        if category in cls._BBQ_CATEGORIES:
            return 'BBQ'
        if category in cls._STEREOSET_CATEGORIES:
            return 'StereoSet'
        return 'unknown'

    @staticmethod
    def _mean_metric_dict(records: list[dict[str, float]]) -> dict[str, float]:
        """Compute mean value per metric key across a list of metric dicts."""
        metric_keys = {key for record in records for key in record}
        return {
            key: fmean(record[key] for record in records if key in record)
            for key in metric_keys
        }

    def _strata_key(self, record: _ExampleEvalRecord) -> tuple[str, ...]:
        """Compose stable tuple key from configured stratification fields."""
        return tuple(record.metadata[field] for field in self._STRATIFY_FIELDS)

    @staticmethod
    def _ci95(values: list[float]) -> float:
        """Compute 95% confidence half-width for a list of values."""
        if len(values) <= 1:
            return 0.0
        return 1.96 * (pstdev(values) / sqrt(len(values)))

    def _metric_uncertainty(self, records: list[dict[str, float]]) -> dict[str, float]:
        """Compute per-metric 95% confidence half-width."""
        metric_keys = {key for record in records for key in record}
        return {
            key: self._ci95([record[key] for record in records if key in record])
            for key in metric_keys
        }

    def _validate_stratified_dimensions(self, rows: list[_StratifiedRow]) -> None:
        """Ensure every row contains exactly the configured dimension keys."""
        expected = set(self._STRATIFY_FIELDS)
        for row in rows:
            row_keys = set(row.dimensions.keys())
            if row_keys != expected:
                raise ValueError(
                    'Stratified row has inconsistent dimensions: '
                    f'expected={sorted(expected)}, got={sorted(row_keys)}'
                )

    def _stratify_records(
        self,
        records: list[_ExampleEvalRecord],
    ) -> list[_StratifiedRow]:
        """Aggregate records by configured stratification keys."""
        sorted_records = sorted(records, key=self._strata_key)
        rows = [
            _StratifiedRow(
                dimensions={field: bucket[0].metadata[field] for field in self._STRATIFY_FIELDS},
                support=len(bucket),
                metrics=self._mean_metric_dict([record.metrics for record in bucket]),
                ci95=self._metric_uncertainty([record.metrics for record in bucket]),
            )
            for _key, grouped in groupby(sorted_records, key=self._strata_key)
            for bucket in [list(grouped)]
        ]
        self._validate_stratified_dimensions(rows)
        return rows

    def _evaluate_single_example(
        self,
        predict_fn: PredictFnAdapter,
        example: dspy.Example,
        extra_metadata: dict[str, Any] | None,
    ) -> _ExampleEvalRecord:
        """Evaluate one example and return metric + metadata record."""
        inputs = example.toDict()
        prediction = predict_fn(**inputs)

        metric_extractors = {
            self._METRIC_NAME_MAP['system_robustness']: system_robustness,
            self._METRIC_NAME_MAP['emergence_rate']: emergence_rate,
            self._METRIC_NAME_MAP['amplification_rate']: amplification_rate,
            self._METRIC_NAME_MAP['propagation_rate']: propagation_rate,
        }
        sample_metrics = {
            metric_name: self._feedback_value(scorer_fn(inputs=inputs, outputs=prediction))
            for metric_name, scorer_fn in metric_extractors.items()
        }
        return _ExampleEvalRecord(
            metrics=sample_metrics,
            metadata=self._extract_metadata(inputs, extra_metadata),
        )

    def _evaluate_deterministic(
        self,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compute metrics directly via scorer functions without LLM judge."""
        predict_fn = PredictFnAdapter(program)
        records: list[_ExampleEvalRecord] = []
        failed_examples: list[dict[str, Any]] = []

        for index, example in enumerate(devset):
            inputs = example.toDict()
            try:
                records.append(self._evaluate_single_example(predict_fn, example, extra_metadata))
            except Exception as error:
                failed_examples.append({
                    'index': index,
                    'error': str(error),
                    'metadata': self._extract_metadata(inputs, extra_metadata),
                })

        if not records:
            raise ValueError(
                'Deterministic evaluation produced no successful examples; '
                f'failed={len(failed_examples)}.'
            )

        overall_metrics = self._mean_metric_dict([record.metrics for record in records])
        stratified_rows = self._stratify_records(records)
        output = _EvaluatorOutput(
            backend=EvaluatorBackend.DETERMINISTIC.value,
            system_robustness=self._extract_system_robustness(overall_metrics),
            genai_evaluation=None,
            detailed_results=[
                {
                    'metrics': record.metrics,
                    'metadata': record.metadata,
                }
                for record in records
            ],
            config=getattr(program, 'config', None),
            genai_metrics=overall_metrics,
            stratified_metrics=[row.as_dict() for row in stratified_rows],
            uncertainty=self._metric_uncertainty([record.metrics for record in records]),
            failure_count=len(failed_examples),
            processed_count=len(records),
            failed_examples=failed_examples,
        )
        return output.as_dict()

    def _evaluate_genai(
        self,
        program: dspy.Module,
        devset: list[dspy.Example],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run MLflow GenAI evaluation and preserve compatibility keys."""
        eval_data = self._to_mlflow_eval_dataset(devset)
        predict_fn = PredictFnAdapter(program)
        genai_result = evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[
                system_robustness,
                emergence_rate,
                amplification_rate,
                propagation_rate,
            ],
        )

        run_meta = self._extract_metadata({}, extra_metadata)
        numeric_metrics = {
            key: float(value)
            for key, value in genai_result.metrics.items()
        }
        stratified_rows = [
            _StratifiedRow(
                dimensions={field: run_meta[field] for field in self._STRATIFY_FIELDS},
                support=len(devset),
                metrics=numeric_metrics,
                ci95=dict.fromkeys(numeric_metrics, 0.0),
            )
        ]
        self._validate_stratified_dimensions(stratified_rows)

        output = _EvaluatorOutput(
            backend=EvaluatorBackend.GENAI.value,
            system_robustness=self._extract_system_robustness(genai_result.metrics),
            genai_evaluation=genai_result,
            detailed_results=None,
            config=getattr(program, 'config', None),
            genai_metrics=genai_result.metrics,
            stratified_metrics=[row.as_dict() for row in stratified_rows],
            uncertainty=dict.fromkeys(numeric_metrics, 0.0),
            failure_count=0,
            processed_count=len(devset),
            failed_examples=[],
        )
        return output.as_dict()

    def __call__(
        self,
        program: dspy.Module,
        devset: list[dspy.Example] | None = None,
        backend: EvaluatorBackend | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate ``program`` and return summary metrics and stratified reports.

        Returns:
            Dictionary with aggregate robustness, backend-specific evaluation,
            and metric maps including metadata-aware stratified aggregates.
        """
        devset = devset or self.devset
        chosen_backend = EvaluatorBackend(backend or self.backend)

        if chosen_backend == EvaluatorBackend.GENAI:
            return self._evaluate_genai(program=program, devset=devset, extra_metadata=metadata)

        return self._evaluate_deterministic(
            program=program,
            devset=devset,
            extra_metadata=metadata,
        )
