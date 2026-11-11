"""Minimal API. Run from backend with: uvicorn main:app --reload."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.papers import router as papers_router
from core.config import settings
from db.database import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare storage and tables at startup; release connections at shutdown."""
    try:
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        await init_db()
        yield
    finally:
        await engine.dispose()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(papers_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "app": settings.APP_NAME}
