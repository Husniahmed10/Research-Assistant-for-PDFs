"""Async database engine, session factory, and model base."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from core.config import settings


engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


Base = declarative_base()


async def init_db() -> None:
    """Create registered model tables; call during application startup."""
    from db import models  # noqa: F401 -- register model tables with Base.metadata

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
