"""Repository layer for unified dataset access.

Provides a small async abstraction over SQLModel queries used by data splitting
and evaluation preparation workflows.
"""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from bias_mitigation.data.schemas.datasets import UnifiedBiasEntry


class UnifiedEntryRepository:
    """Async repository for reading ``UnifiedBiasEntry`` records.

    The repository owns an async SQLAlchemy engine/session factory and exposes
    read operations used by downstream pipeline stages.
    """

    def __init__(self, db_url: str):
        """Initialize async engine/session factory.

        Args:
            db_url: SQLAlchemy async database URL.
        """
        self.engine = create_async_engine(db_url)
        self.session_maker = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def fetch_all(self) -> Sequence[UnifiedBiasEntry]:
        """Return all unified entries currently stored in the database.

        Returns:
            Ordered sequence of ``UnifiedBiasEntry`` rows as returned by the
            database query execution.
        """
        async with self.session_maker() as session:
            stmt = select(UnifiedBiasEntry)
            result = await session.exec(stmt)
            return result.all()
