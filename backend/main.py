"""Research assistant API. Run from backend: uvicorn main:app --reload."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool

from core.config import settings
from tracing.langsmith_setup import setup_langsmith

# Set SDK environment before importing routes, which import the compiled graph.
setup_langsmith()

from core.exceptions import (
    ErrorHandlingRoute, internal_error_handler, http_error_handler, validation_error_handler,
)
from db.database import engine, init_db
from services.pinecone_client import init_pinecone_index, load_bm25_encoders
from tracing.logfire_setup import setup_logfire


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize tables, Pinecone, BM25 encoders and uploads, in that order."""
    app.state.pinecone_index = None
    try:
        await init_db()
        app.state.pinecone_index = await run_in_threadpool(init_pinecone_index)
        await run_in_threadpool(load_bm25_encoders)
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        yield
    except Exception as exc:
        logfire.error("Application lifecycle failed", error_type=type(exc).__name__)
        raise
    finally:
        app.state.pinecone_index = None
        await engine.dispose()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.router.route_class = ErrorHandlingRoute
app.add_exception_handler(Exception, internal_error_handler)
app.add_exception_handler(StarletteHTTPException, http_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Instrumentation installs middleware, so it must precede ASGI lifespan startup.
setup_logfire(app)

from api.routes.papers import router as papers_router
from api.routes.chat import router as chat_router
from api.routes.evaluation import router as evaluation_router
from api.routes.health import router as health_router

# Register routes once, before serving requests (never per lifespan invocation).
app.include_router(papers_router)
app.include_router(chat_router)
app.include_router(evaluation_router)
app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "app": settings.APP_NAME}
