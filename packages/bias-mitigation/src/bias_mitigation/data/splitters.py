"""Data split strategies for reproducible train/dev generation."""
import random
from abc import ABC, abstractmethod
from itertools import chain, groupby

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
            )
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

        def split_group(group: list[UnifiedBiasEntry]) -> tuple[list[SplitRecord], list[SplitRecord]]:
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
