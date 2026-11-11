"""Minimal API. Run from backend with: uvicorn main:app --reload."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from api.routes.papers import router as papers_router
from core.config import settings
from db.database import engine, init_db
from services.pinecone_client import init_pinecone_index


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare storage, tables and vector index before accepting requests."""
    app.state.pinecone_index = None
    try:
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        await init_db()
        app.state.pinecone_index = await run_in_threadpool(init_pinecone_index)
        yield
    finally:
        app.state.pinecone_index = None
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
