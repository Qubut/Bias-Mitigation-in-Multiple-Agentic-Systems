"""Dataset loading helpers for DSPy evaluation inputs.

This module loads serialized split files and reconstructs ``dspy.Example``
objects with stable input-key configuration.
"""

import json
import pathlib

from dspy import Example


def load_dspy_dataset(
    path: str | pathlib.Path,
    input_keys: tuple[str, ...] = (
        'context',
        'question',
        'ans0',
        'ans1',
        'ans2',
        'category',
        'stereotyped_groups',
    ),
) -> list[Example]:
    """Load one JSON split file and return DSPy ``Example`` records.

    Args:
        path: Path to split JSON (e.g. ``datasets/splits/devset.json``).
        input_keys: Tuple of keys treated as model inputs during evaluation.

    Returns:
        List of reconstructed ``dspy.Example`` objects.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Notes:
        Supports both flattened records and nested records containing
        ``example``/``dataset_metadata`` payloads.
    """
    file_path = pathlib.Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {file_path}')

    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    return [
        Example(
            **(
                item
                if 'example' not in item
                else {**item.get('example', {}), **item.get('dataset_metadata', {})}
            )
        ).with_inputs(*input_keys)
        for item in data
    ]


def load_splits(
    base_dir: str | pathlib.Path = 'datasets/splits',
) -> tuple[list[Example], list[Example]]:
    """Load train/dev split files from a base directory.

    Args:
        base_dir: Directory containing ``trainset.json`` and ``devset.json``.

    Returns:
        Tuple of ``(trainset, devset)`` examples.
    """
    dir_path = pathlib.Path(base_dir)
    train_path = dir_path / 'trainset.json'
    dev_path = dir_path / 'devset.json'

    trainset = load_dspy_dataset(train_path)
    devset = load_dspy_dataset(dev_path)

    return trainset, devset
