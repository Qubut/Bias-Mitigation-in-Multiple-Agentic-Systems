"""Per-sample metadata derivation for the MAS evaluator.

Every evaluation sample needs a normalized metadata envelope that drives
stratification, MLflow tagging, mem0 correlation, and the analysis
schema descriptor.  This module owns that derivation so the rest of the
evaluator can stay focused on orchestration: predict, score, stream,
aggregate.

The :class:`MetadataExtractor` is intentionally state-light — it carries
only the run-wide defaults (``run_metadata``) plus a few class
constants — and exposes:

* :meth:`extract` — merge run defaults, per-call overrides, and per-sample
  inputs into a validated :class:`_EvaluationMetadata`.
* :meth:`resolve_sample_id` — synthesize a stable sample id so re-runs
  correlate.
* :meth:`build_agent_turn_rows` — flatten the MAS debate history into the
  long-format rows used by fairness notebooks.
* :meth:`analysis_schema` — emit the schema descriptor consumed by
  downstream analysis tooling.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, ClassVar

import dspy
from returns.result import Failure, Success, safe

from ..metrics import build_agent_turn_bias_series
from .models import _AgentTurnRow, _EvaluationMetadata, _RoundMetricRow, _SampleOutcomeRow


class MetadataExtractor:
    """Build the per-sample metadata envelope used everywhere else.

    The evaluator delegates every "what does this sample belong to?"
    question to an instance of this class so the answers stay consistent
    between stratification, MLflow tagging, mem0 correlation, and the
    streamed JSONL artefacts.

    Attributes:
        run_metadata: Static defaults supplied once by the evaluator's
            caller (run id, protocol, intervention, …).  Merged into
            every per-sample metadata payload below any per-call
            overrides but above pure ``"unknown"`` fallbacks.
        STRATIFY_FIELDS: Ordered tuple of metadata keys that define each
            stratum in the fairness report.  Treated as the single source
            of truth — the aggregator and the analysis-schema descriptor
            both read from here.
    """

    STRATIFY_FIELDS: ClassVar[tuple[str, ...]] = (
        'dataset_name',
        'dataset_source',
        'stereoset_type',
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

    def __init__(self, run_metadata: dict[str, Any] | None = None) -> None:
        """Configure the extractor with the run-wide metadata defaults.

        Args:
            run_metadata: Static metadata to merge into every per-sample
                payload (e.g. ``run_id``, ``protocol``, ``intervention``).
                ``None`` is treated as an empty mapping.
        """
        self.run_metadata = run_metadata or {}

    def extract(
        self,
        inputs: dict[str, Any],
        extra_metadata: dict[str, Any] | None,
    ) -> _EvaluationMetadata:
        """Derive a normalized metadata record for a single sample.

        Combines, in order of decreasing priority, fields supplied on
        the dataset example itself (``inputs``), the per-call
        ``extra_metadata`` overrides, and the evaluator's
        ``run_metadata`` defaults. Dataset, category, and stereoset
        type are inferred when not explicitly provided so heterogeneous
        BBQ/StereoSet inputs share a uniform schema.

        Args:
            inputs: Raw input dictionary from a ``dspy.Example``.
            extra_metadata: Optional per-evaluation overrides.

        Returns:
            A validated :class:`_EvaluationMetadata` instance used for
            stratification, MLflow tags, and stream events.
        """
        merged: dict[str, Any] = {}
        merged.update(self.run_metadata)
        if extra_metadata:
            merged.update(extra_metadata)

        category = inputs.get('category') or merged.get('category') or 'unknown'
        dataset_source = inputs.get('source') or merged.get('dataset_source')
        stereoset_type = (
            inputs.get('original_type') or inputs.get('subcategory') or merged.get('stereoset_type')
        )

        dataset_name = (
            inputs.get('dataset_name')
            or inputs.get('dataset')
            or dataset_source
            or merged.get('dataset_name')
            or self.infer_dataset_name(str(category))
        )

        metadata_payload = {
            'dataset_name': str(dataset_name),
            'dataset_source': str(dataset_source or dataset_name),
            'stereoset_type': str(stereoset_type or 'none'),
            'category': str(category),
            'protocol': str(merged.get('protocol', 'unknown')),
            'llm_models': str(merged.get('llm_models', 'unknown')),
            'model_names': str(merged.get('model_names', 'unknown')),
            'intervention': str(merged.get('intervention', 'unknown')),
            'num_agents': str(merged.get('num_agents', 'unknown')),
            'rounds': str(merged.get('rounds', 'unknown')),
            'split': str(merged.get('split', 'unknown')),
            'seed': str(merged.get('seed', 'unknown')),
            'run_id': str(merged.get('run_id', 'unknown')),
            'agent_model_map': str(merged.get('agent_model_map', '{}')),
        }
        return _EvaluationMetadata.model_validate(metadata_payload)

    @staticmethod
    def parse_agent_model_map(raw_map: str) -> dict[str, str]:
        """Safely parse the serialized ``agent_model_map`` metadata field.

        Metadata is stored as a JSON string so it survives Pydantic
        serialization and MLflow tagging. This helper decodes it into
        an ``agent_name -> model_id`` mapping, falling back to an empty
        dict on any parse or type error.

        Args:
            raw_map: JSON-encoded mapping (possibly malformed).

        Returns:
            A ``dict[str, str]`` mapping agent names to their model ids,
            or an empty dict if decoding fails.
        """

        @safe
        def load(s: str) -> Any:
            return json.loads(s)

        return (
            load(raw_map)
            .bind(lambda p: Success(p) if isinstance(p, dict) else Failure(TypeError('not a dict')))
            .map(lambda p: {str(k): str(v) for k, v in p.items()})
            .value_or({})
        )

    @staticmethod
    def canonical_sample_text(inputs: dict[str, Any]) -> str:
        """Construct a deterministic textual fingerprint for a sample.

        The fingerprint concatenates the context, question, answer
        options, and category so semantically identical samples receive
        the same hash even when their position in the dataset shifts.
        This is what stabilises generated sample ids across reruns.

        Args:
            inputs: Raw inputs of a ``dspy.Example``.

        Returns:
            A newline-joined canonical string ready for hashing.
        """
        options = [
            str(inputs.get('ans0', '')),
            str(inputs.get('ans1', '')),
            str(inputs.get('ans2', '')),
        ]
        return '\n'.join([
            str(inputs.get('context', '')),
            str(inputs.get('question', '')),
            *options,
            str(inputs.get('category', 'unknown')),
        ])

    def resolve_sample_id(self, inputs: dict[str, Any], example_index: int) -> str:
        """Pick or synthesize a stable sample identifier.

        Preference order:

        1. An explicit ``sample_id`` or ``id`` field on the example.
        2. A deterministic ``sample-<index>-<hash>`` id built from the
           canonical sample text.

        Stable ids are important because they correlate stream events,
        mem0 memory entries, and post-hoc analysis joins across reruns.

        Args:
            inputs: Raw inputs of a ``dspy.Example``.
            example_index: Position of the example in the devset
                (already offset by ``index_offset``).

        Returns:
            A string sample identifier.
        """
        direct_id = inputs.get('sample_id') or inputs.get('id')
        if isinstance(direct_id, str) and direct_id.strip():
            return direct_id.strip()
        if isinstance(direct_id, int):
            return str(direct_id)

        digest = sha256(self.canonical_sample_text(inputs).encode('utf-8')).hexdigest()[:16]
        return f'sample-{example_index}-{digest}'

    def build_agent_turn_rows(
        self,
        inputs: dict[str, Any],
        prediction: dspy.Prediction,
        sample_id: str,
        example_index: int,
        metadata: _EvaluationMetadata,
        sample_run_id: str | None,
    ) -> list[_AgentTurnRow]:
        """Materialise per-agent-turn bias rows for one sample.

        Walks the agent debate ``history`` recorded by the MAS program,
        runs the per-turn bias attribution defined in
        :mod:`bias_mitigation.mas.metrics`, and produces the long-format
        rows consumed by downstream notebooks and the streaming sinks.

        Args:
            inputs: Raw sample inputs.
            prediction: DSPy ``Prediction`` returned by the MAS program.
                Expected to expose a ``history`` mapping of
                ``agent_name -> list[turn]``.
            sample_id: Stable sample id (see :meth:`resolve_sample_id`).
            example_index: Position of the sample in the devset.
            metadata: Validated metadata used to tag every row.
            sample_run_id: Optional sample-scoped MLflow/tracing run id.

        Returns:
            A list of :class:`_AgentTurnRow` instances, one per agent
            turn observed in the prediction history.
        """
        model_map = self.parse_agent_model_map(metadata.agent_model_map)
        base_turn_rows = build_agent_turn_bias_series(
            inputs=inputs,
            outputs=prediction,
            agent_model_map=model_map,
        )
        history = getattr(prediction, 'history', {})
        final_turn_index_by_agent = {
            str(agent_name): max(len(turns) - 1, 0) for agent_name, turns in history.items()
        }
        return _AgentTurnRow.from_series(
            base_turn_rows=base_turn_rows,
            final_turn_index_by_agent=final_turn_index_by_agent,
            sample_id=sample_id,
            sample_run_id=sample_run_id,
            example_index=example_index,
            metadata=metadata,
        )

    def analysis_schema(self) -> dict[str, Any]:
        """Return the schema descriptor embedded in evaluator output.

        Downstream notebooks rely on this descriptor to know which
        columns to expect in the sample, agent-turn, and round-metric
        artefacts and which fields define the stratification grid. The
        ``artifact_schema_version`` lets consumers detect breaking
        changes when the schema evolves.

        Returns:
            A dict describing artefact columns and stratify fields.
        """
        return {
            'artifact_schema_version': '1.1.0',
            'sample_outcomes_columns': _SampleOutcomeRow.analysis_columns(),
            'agent_turns_columns': _AgentTurnRow.analysis_columns(),
            'round_metrics_columns': _RoundMetricRow.analysis_columns(),
            'stratify_fields': list(self.STRATIFY_FIELDS),
        }

    @classmethod
    def infer_dataset_name(cls, category: str) -> str:
        """Guess the source dataset from a sample's category label.

        Useful when datasets are merged at the dataloader level and the
        original ``dataset_name`` field is lost. Categories are matched
        against the known BBQ and StereoSet category sets.

        Args:
            category: Sample category label (e.g. ``"Age"``,
                ``"intersentence"``).

        Returns:
            ``"BBQ"``, ``"StereoSet"``, or ``"unknown"``.
        """
        if category in cls._BBQ_CATEGORIES:
            return 'BBQ'
        if category in cls._STEREOSET_CATEGORIES:
            return 'StereoSet'
        return 'unknown'
