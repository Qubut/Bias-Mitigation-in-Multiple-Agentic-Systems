# bias_mitigation/data/unify_datasets.py
import asyncio
import hashlib
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import batched, groupby
from typing import Any, TypeVar

import click
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload
from sqlmodel import SQLModel, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from bias_mitigation.data.schemas.datasets import (
    BBQ,
    StereoSet,
    UnifiedBiasEntry,
)

BIAS_TO_CATEGORY_MAP = {
    'gender': 'Gender_identity',
    'race': 'Race_ethnicity',
    'religion': 'Religion',
    'profession': 'Profession',
}


class UnificationConfig(BaseSettings):
    model_config = SettingsConfigDict(extra='forbid', env_prefix='UNIFY_')

    random_seed: int = 42
    bbq_target_total: int = 8_600
    stereoset_target_total: int = 8_600
    balance_categories: bool = True  # new: stratify BBQ + StereoSet


T = TypeVar('T', bound=SQLModel)


class TransformerStrategy[T: SQLModel](ABC):
    @abstractmethod
    def transform(self, item: T) -> UnifiedBiasEntry | None:
        """Base implementation strictly mapping abstract structure."""


class BBQTransformer(TransformerStrategy[BBQ]):
    def transform(self, item: BBQ) -> UnifiedBiasEntry | None:
        """Transform BBQ context variables bound structurally mapping."""
        if not BBQTransformer._is_ambiguous(item):
            return None
        return UnifiedBiasEntry(
            source='BBQ',
            example_id=item.example_id,
            category=item.category,
            context=item.context,
            question=item.question,
            ans0=item.ans0,
            ans1=item.ans1,
            ans2=item.ans2,
            label=item.label,
            stereotyped_groups=item.stereotyped_groups,
            additional_metadata=item.additional_metadata,
        )

    @staticmethod
    def _is_ambiguous(bbq: BBQ) -> bool:
        if not bbq.answers or bbq.label >= len(bbq.answers):
            return False
        correct = bbq.answers[bbq.label]
        text_lower = correct.text.lower()
        tag_lower = correct.tag.lower()
        return 'unknown' in text_lower or 'not enough' in text_lower or tag_lower == 'unknown'


class StereoSetTransformer(TransformerStrategy[StereoSet]):
    def transform(self, item: StereoSet) -> UnifiedBiasEntry | None:
        """Transform StereoSet sentences randomizing mapping indexes."""
        category = BIAS_TO_CATEGORY_MAP.get(item.bias_type.lower())
        if category is None:
            return None

        stereotype = next(
            (s.sentence for s in item.sentences if s.gold_label == 'stereotype'), None
        )
        anti = next((s.sentence for s in item.sentences if s.gold_label == 'anti-stereotype'), None)
        if not stereotype or not anti:
            stereotype = item.sentences[0].sentence if item.sentences else ''
            anti = item.sentences[1].sentence if len(item.sentences) > 1 else ''

        example_id = int(hashlib.sha256(item.id.encode('utf-8')).hexdigest()[:8], 16)

        question = f'Which of the following best completes the context regarding {item.target}?'
        if item.context:
            question = f'{item.context} {question}'

        choices = [stereotype, anti, 'Unknown']
        random.shuffle(choices)
        ans0, ans1, ans2 = choices
        label = choices.index('Unknown')

        return UnifiedBiasEntry(
            source='StereoSet',
            example_id=example_id,
            category=category,
            context=item.context,
            question=question,
            ans0=ans0,
            ans1=ans1,
            ans2=ans2,
            label=label,
            stereotyped_groups=[item.target],
            additional_metadata={
                'original_type': item.type,
                'bias_type': item.bias_type,
                'target': item.target,
                'original_id': item.id,
                'stereotype_index': choices.index(stereotype),
                'anti_stereotype_index': choices.index(anti)
            },
        )


class TransformerFactory:
    @staticmethod
    def get(source: str) -> TransformerStrategy[Any]:
        """Factory method to get a transformer by source name."""
        if source == 'BBQ':
            return BBQTransformer()
        if source == 'StereoSet':
            return StereoSetTransformer()
        raise ValueError(f'Unknown source: {source}')


class CategoryBalancer:
    """Strategy for balanced sampling per subclass - prevents dominance."""

    @staticmethod
    def balance(
        items: list[UnifiedBiasEntry], target_total: int, *, balance: bool
    ) -> list[UnifiedBiasEntry]:
        """Balance items stratifying across available bias categories."""
        if not items:
            return []

        if not balance:
            return random.sample(items, min(target_total, len(items)))

        # Declarative grouping and stratification mapping
        sorted_items = sorted(items, key=lambda x: x.category)
        groups = {k: list(v) for k, v in groupby(sorted_items, key=lambda x: x.category)}
        per_cat = max(1, target_total // len(groups))

        # Core distributed sampling
        balanced = [
            sampled
            for cat_items in groups.values()
            for sampled in random.sample(cat_items, min(per_cat, len(cat_items)))
        ]

        # Fill remainder dynamically to strictly match target size without loop accumulation
        if (remainder := target_total - len(balanced)) > 0:
            remaining_pool = [i for i in items if i not in balanced]
            balanced.extend(random.sample(remaining_pool, min(remainder, len(remaining_pool))))

        logger.info(f'Balanced {len(groups)} categories → {len(balanced)} items (target {target_total})')
        return balanced


class UnifiedRepository:
    """Repository pattern: isolates all database operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def count(self) -> int:
        """Count exist unified records."""
        async with self.session_factory() as s:
            result = await s.exec(select(func.count()).select_from(UnifiedBiasEntry))
            return result.one_or_none() or 0

    async def truncate(self) -> None:
        """Truncate the table."""
        async with self.session_factory.begin() as conn:
            await conn.exec(delete(UnifiedBiasEntry))

    async def insert_batch(self, batch: Sequence[UnifiedBiasEntry]) -> int:
        """Insert a batch of records."""
        async with self.session_factory() as session, session.begin():
            for model in batch:
                session.add(model)
        return len(batch)


async def orchestrate_unification(
    engine: AsyncEngine, cfg: UnificationConfig, *, force: bool
) -> dict[str, int]:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    repo = UnifiedRepository(session_factory)

    # Idempotency (enterprise safety)
    if (existing := await repo.count()) > 0:
        if not force:
            logger.info(f'Unified table already has {existing} entries - skipping')
            return {'already_exists': existing}
        await repo.truncate()
        logger.info('Force truncation completed')

    # Extract
    async with session_factory() as s:
        bbq_list = (await s.exec(select(BBQ).options(selectinload(BBQ.answers)))).all()  # type: ignore[arg-type]
        ss_list = (await s.exec(select(StereoSet).options(selectinload(StereoSet.sentences)))).all()  # type: ignore[arg-type]

    # Transform via Strategy + Factory
    factory = TransformerFactory()
    bbq_transformed = [factory.get('BBQ').transform(b) for b in bbq_list]
    ss_transformed = [factory.get('StereoSet').transform(s) for s in ss_list]

    unified_bbq = [u for u in bbq_transformed if u is not None]
    unified_ss = [u for u in ss_transformed if u is not None]

    # Deterministic sampling to EXACT paper sizes (best practice over paper's GPT inference)
    random.seed(cfg.random_seed)
    unified_bbq = random.sample(unified_bbq, min(cfg.bbq_target_total, len(unified_bbq)))
    unified_ss = random.sample(unified_ss, min(cfg.stereoset_target_total, len(unified_ss)))

    unified_list = unified_bbq + unified_ss
    logger.info(
        f'Sampled → BBQ ambiguous: {len(unified_bbq)} | StereoSet: {len(unified_ss)} | Total: {len(unified_list)}'
    )

    # Insert (same batched pattern as original ingest)
    chunk_size = 1000
    batches = list(batched(unified_list, chunk_size))
    totals = [await repo.insert_batch(batch) for batch in batches]

    total = sum(totals)
    logger.info(f'✅ UnifiedBiasEntry populated with {total} MCQ samples')
    return {'unified': total, 'bbq_ambiguous': len(unified_bbq), 'stereoset': len(unified_ss)}


async def unify_async(db_url: str, *, force: bool) -> None:
    engine = create_async_engine(db_url, echo=False)
    cfg = UnificationConfig()
    result = await orchestrate_unification(engine, cfg, force=force)
    logger.info(f'Unification complete: {result}')


@click.command()
@click.option(
    '--db-url', default=lambda: os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./datasets.db')
)
@click.option('--force', '-f', is_flag=True)
def run(db_url: str, force: bool) -> None:  # noqa: FBT001
    if force:
        logger.info('Force mode: will truncate unified table')
    asyncio.run(unify_async(db_url, force=force))


if __name__ == '__main__':
    run()
