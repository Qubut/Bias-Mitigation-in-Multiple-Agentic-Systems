"""Async SQLModel repository for unified dataset access."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from bias_mitigation.data.schemas.datasets import UnifiedBiasEntry


class UnifiedEntryRepository:
    """Async repository for reading ``UnifiedBiasEntry`` records."""

    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url)
        self.session_maker = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def fetch_all(self) -> Sequence[UnifiedBiasEntry]:
        async with self.session_maker() as session:
            stmt = select(UnifiedBiasEntry)
            result = await session.exec(stmt)
            return result.all()
