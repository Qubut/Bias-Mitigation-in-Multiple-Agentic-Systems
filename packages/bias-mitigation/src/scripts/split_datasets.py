"""CLI entrypoint for stratified train/dev split generation."""

import asyncio
import json
import os
import pathlib
from typing import Any

import click
from loguru import logger

from bias_mitigation.data.repository import UnifiedEntryRepository
from bias_mitigation.data.splitters import StratifiedCategorySplitter


@click.command()
@click.option(
    '--db-url',
    default=lambda: os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./datasets.db'),
    help='Database URL to pull unified dataset from.',
)
@click.option(
    '--train-ratio', default=0.5, help='Percentage of data to use for training (default: 0.5)'
)
@click.option('--seed', default=42, help='Random seed for consistent splits (default: 42)')
@click.option(
    '--output-dir', default='./datasets/splits', help='Output directory for dspy Example files'
)
def run(db_url: str, train_ratio: float, seed: int, output_dir: str):
    """Run asynchronous stratified split with CLI-provided parameters."""
    asyncio.run(split_data(db_url, train_ratio, seed, output_dir))


async def split_data(db_url: str, train_ratio: float, seed: int, output_dir: str):
    """Fetch unified entries, split by category, and write JSON outputs."""
    logger.info(f'Starting stratified data splitting (Train Ratio: {train_ratio}, Seed: {seed})')

    repository = UnifiedEntryRepository(db_url)

    entries = await repository.fetch_all()

    entries_list = list(entries)
    logger.info(f'Loaded {len(entries_list)} entries from the unified dataset.')

    splitter = StratifiedCategorySplitter(train_ratio=train_ratio, seed=seed)
    trainset, devset = splitter.split(entries_list)

    logger.info(f'Split generated: {len(trainset)} train, {len(devset)} dev examples.')

    # Ensure output directory exists
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    # DSPy Examples can be serialized to JSON safely
    train_path = out_path / 'trainset.json'
    dev_path = out_path / 'devset.json'

    def serialize_examples(examples: list[Any], path: pathlib.Path) -> None:
        """Serialize DSPy examples as JSON using ``model_dump`` output."""
        with path.open('w', encoding='utf-8') as f:
            # We convert Pydantic models to dict using model_dump
            data = [e.model_dump() for e in examples]
            json.dump(data, f, indent=2)

    serialize_examples(trainset, train_path)
    serialize_examples(devset, dev_path)
    logger.info(f'Saved splits to {output_dir}')


if __name__ == '__main__':
    run()
