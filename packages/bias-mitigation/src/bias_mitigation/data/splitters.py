"""Data split strategies for reproducible train/dev generation."""

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain, groupby
from math import floor

from dspy import Example

from bias_mitigation.data.schemas.datasets import (
    DatasetExample,
    DatasetMetadata,
    SplitRecord,
    UnifiedBiasEntry,
)


class AbstractSplitStrategy(ABC):
    """Abstract contract for split strategy implementations."""

    @abstractmethod
    def split(self, data: list[UnifiedBiasEntry]) -> tuple[list[SplitRecord], list[SplitRecord]]:
        """Split data into a (trainset, devset) tuple of dictionary records."""


class StratifiedCategorySplitter(AbstractSplitStrategy):
    """Stratified splitter over category/source/type groups."""

    def __init__(self, train_ratio: float = 0.5, seed: int = 42):
        """Initialize split ratio and deterministic RNG state."""
        self.train_ratio = train_ratio
        self.seed = seed

        self.rng = random.Random(seed)

    def _to_record(self, entry: UnifiedBiasEntry) -> SplitRecord:
        """Convert one unified entry into the serialized split record format."""
        return SplitRecord(
            dataset_metadata=DatasetMetadata(
                source=entry.source,
                category=entry.category,
                subcategory=entry.additional_metadata.get('subcategory'),
                original_type=entry.additional_metadata.get('original_type'),
                context_condition=entry.additional_metadata.get('context_condition'),
            ),
            example=DatasetExample(
                context=entry.context,
                question=entry.question,
                ans0=entry.ans0,
                ans1=entry.ans1,
                ans2=entry.ans2,
                label=entry.label,
                sample_id=f'{entry.source}:{entry.example_id}',
                entry_id=entry.id,
                source=entry.source,
            ),
        )

    def split(self, data: list[UnifiedBiasEntry]) -> tuple[list[SplitRecord], list[SplitRecord]]:
        # Sort to ensure stable groupby (stratifying via category + source + internal data type)
        """Split entries into train/dev records using grouped stratification."""

        def get_group_key(item: UnifiedBiasEntry) -> tuple[str, str, str]:
            # Extrapolate 'intra'/'inter' sentence structure for StereoSet balancing
            """Build grouping key for stratified split buckets."""
            sub_type = item.additional_metadata.get('original_type', 'none')
            return (item.category, item.source, sub_type)

        sorted_data = sorted(data, key=get_group_key)
        groups = [list(group) for _, group in groupby(sorted_data, key=get_group_key)]

        def split_group(
            group: list[UnifiedBiasEntry],
        ) -> tuple[list[SplitRecord], list[SplitRecord]]:
            """Split one group into train/dev subsets using deterministic shuffling."""
            shuffled = self.rng.sample(group, len(group))
            split_idx = int(len(shuffled) * self.train_ratio)
            train_items = [self._to_record(item) for item in shuffled[:split_idx]]
            dev_items = [self._to_record(item) for item in shuffled[split_idx:]]
            return train_items, dev_items

        group_splits = (split_group(group) for group in groups)
        train_groups, dev_groups = zip(*group_splits, strict=False)

        trainset = list(chain.from_iterable(train_groups))
        devset = list(chain.from_iterable(dev_groups))

        # Global shuffle so the final sets aren't clustered by category/source blocks
        trainset = self.rng.sample(trainset, len(trainset))
        devset = self.rng.sample(devset, len(devset))

        return trainset, devset


def _to_example_dict(example: Example) -> dict[str, object]:
    return example.toDict() if hasattr(example, 'toDict') else dict(example.__dict__)


def _normalize_stratum_value(raw_value: object) -> str:
    match raw_value:
        case str() as text if text.strip():
            return text.strip()
        case None:
            return 'unknown'
        case _:
            return str(raw_value)


def _stratum_key(example: Example) -> tuple[str, str]:
    payload = _to_example_dict(example)
    return (
        _normalize_stratum_value(payload.get('category')),
        _normalize_stratum_value(payload.get('question_polarity')),
    )


def _group_examples(examples: Sequence[Example]) -> list[tuple[tuple[str, str], list[Example]]]:
    grouped_examples: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        grouped_examples[_stratum_key(example)].append(example)
    return [(key, grouped_examples[key]) for key in sorted(grouped_examples)]


def _allocate_stratified_counts(
    group_sizes: dict[tuple[str, str], int], subset_size: int, total_examples: int
) -> dict[tuple[str, str], int]:
    target_share = {key: (subset_size * size) / total_examples for key, size in group_sizes.items()}
    allocations = {key: min(size, floor(target_share[key])) for key, size in group_sizes.items()}
    remaining = subset_size - sum(allocations.values())
    ranked_keys = sorted(
        group_sizes,
        key=lambda key: (
            target_share[key] - allocations[key],
            group_sizes[key],
            key,
        ),
        reverse=True,
    )

    for key in ranked_keys:
        if remaining <= 0:
            break
        if allocations[key] < group_sizes[key]:
            allocations[key] += 1
            remaining -= 1

    return allocations


def select_stratified_subset(
    examples: Sequence[Example],
    subset_size: int,
    seed: int = 42,
) -> list[Example]:
    """Select a deterministic stratified subset across key evaluation dimensions.

    Strata are defined as ``(category, question_polarity)`` with a stable
    ``'unknown'`` fallback for missing values.
    """
    total_examples = len(examples)
    match (total_examples, subset_size):
        case (0, _):
            return []
        case (_, n) if n <= 0 or n >= total_examples:
            return list(examples)
        case _:
            pass

    rng = random.Random(seed)
    ordered_groups = _group_examples(examples)
    group_sizes = {key: len(group) for key, group in ordered_groups}
    allocations = _allocate_stratified_counts(group_sizes, subset_size, total_examples)

    subset = list(
        chain.from_iterable(
            rng.sample(group, allocations[key])
            for key, group in ordered_groups
            if allocations[key] > 0
        )
    )

    return rng.sample(subset, len(subset))
