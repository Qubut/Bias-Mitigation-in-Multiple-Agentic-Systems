"""Dataset loading and MLflow dataset-tracking helpers."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd
from dspy import Example
from mlflow.data import from_pandas  # type: ignore[attr-defined]  # present at runtime in mlflow 2.x
from mlflow.data.dataset_source import DatasetSource
from mlflow.entities import DatasetInput

from bias_mitigation.data.loaders import load_splits


@dataclass(frozen=True)
class MetadataExtractionStrategy:
    """Extract normalized dataset tags for MLflow dashboard filtering."""

    def extract_tags(self, example: Example) -> dict[str, str]:
        """Build a stable tag dictionary from one dataset example."""
        ex_dict = example.toDict() if hasattr(example, 'toDict') else dict(example.__dict__)
        return {
            'dataset.source': ex_dict.get('source', 'BBQ+StereoSet'),
            'dataset.category': ex_dict.get('category', 'unknown'),
            'dataset.original_type': ex_dict.get(
                'original_type', ex_dict.get('subcategory', 'none')
            ),  # inter/intra
            'dataset.split': 'dev',
        }


class UnifiedBiasToMLflowAdapter:
    """Convert split examples into MLflow ``DatasetInput`` objects."""

    @staticmethod
    def to_pandas(examples: Sequence[Example]) -> pd.DataFrame:
        """Convert DSPy examples into a normalized pandas DataFrame."""
        if not examples:
            return pd.DataFrame()
        rows = [ex.toDict() if hasattr(ex, 'toDict') else dict(ex.__dict__) for ex in examples]
        df = pd.DataFrame.from_records(rows)
        # Prevent integer schema warning: cast potential integer columns to float64
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        if not int_cols.empty:
            df[int_cols] = df[int_cols].astype('float64')
        return df

    @staticmethod
    def adapt(examples: Sequence[Example], name: str, version: str) -> DatasetInput:
        """Create one MLflow dataset input object from split examples."""
        df = UnifiedBiasToMLflowAdapter.to_pandas(examples)
        if df.empty:
            raise ValueError('Cannot create MLflow dataset from empty examples')

        source = DatasetSource.from_dict({
            'type': 'unified_bias',
            'version': version,  # stored in source metadata
        })

        # Correct MLflow 2026 API: NO 'version' kwarg on from_pandas
        dataset = from_pandas(
            df=df,
            name=name,  # name carries versioning intent
            source=source,
        )

        if examples:
            dataset.tags = MetadataExtractionStrategy().extract_tags(examples[0])

        return cast(DatasetInput, dataset)


class DatasetTrackerFacade:
    """Facade for split-specific dataset tracking behavior."""

    def __init__(self, *, is_dev: bool = True):
        """Configure dataset naming and context for one split type."""
        self.is_dev = is_dev
        self.name = f'bias_mas_{"dev" if is_dev else "train"}set'
        self.context = 'evaluation' if is_dev else 'training'

    def track(self, examples: Sequence[Example], version: str = 'v1.0') -> DatasetInput:
        """Track one dataset split and return MLflow dataset metadata."""
        return UnifiedBiasToMLflowAdapter.adapt(examples, self.name, version)


class DatasetTrackerFactory:
    """Factory for creating split-specific dataset tracker facades."""

    @staticmethod
    def get(*, is_dev: bool = True) -> DatasetTrackerFacade:
        """Return a dataset tracker configured for train or dev split."""
        return DatasetTrackerFacade(is_dev=is_dev)


def load_and_track_splits(
    base_dir: str = 'datasets/splits',
    version: str = 'v1.0',
) -> tuple[list[Example], list[Example], DatasetInput, DatasetInput]:
    """Load train/dev splits and produce corresponding MLflow dataset inputs."""
    train_examples, dev_examples = load_splits(base_dir)

    train_ds = DatasetTrackerFactory.get(is_dev=False).track(train_examples, version)
    dev_ds = DatasetTrackerFactory.get(is_dev=True).track(dev_examples, version)

    return train_examples, dev_examples, train_ds, dev_ds
