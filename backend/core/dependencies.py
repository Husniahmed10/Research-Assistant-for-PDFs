"""Shared dependencies for FastAPI routes.

Usage::

    db: AsyncSession = Depends(get_db)
    config: Settings = Depends(get_settings)
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, settings
from db.database import AsyncSessionLocal


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield one session per request and close it even if the request fails.

    Callers explicitly commit successful writes. Closing the session rolls back
    any transaction that was left uncommitted.
    """
    async with AsyncSessionLocal() as session:
        yield session


def get_settings() -> Settings:
    """Return the shared application settings instance."""
    return settings
