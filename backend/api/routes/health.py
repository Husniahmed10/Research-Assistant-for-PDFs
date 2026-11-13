"""Live readiness checks for SQL, vector storage and the configured gateway."""

import asyncio
from datetime import datetime, timezone

import logfire
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from core.dependencies import get_db
from core.exceptions import ErrorHandlingRoute
from services import portkey_client

router = APIRouter(prefix="/api/v1/health", tags=["Health"], route_class=ErrorHandlingRoute)
_CHECK_TIMEOUT_SECONDS = 10.0


async def _probe(name, operation) -> str:
    try:
        await asyncio.wait_for(operation(), timeout=_CHECK_TIMEOUT_SECONDS)
        return "ok"
    except Exception as exc:
        logfire.warn("Dependency health check failed", dependency=name, error_type=type(exc).__name__)
        return "error"


async def check_db(db: AsyncSession) -> str:
    async def ping():
        await db.execute(text("SELECT 1"))
    return await _probe("db", ping)


async def check_pinecone(index) -> str:
    async def ping():
        if index is None:
            raise RuntimeError("Vector index is not initialized")
        await run_in_threadpool(index.describe_index_stats, _request_timeout=_CHECK_TIMEOUT_SECONDS)
    return await _probe("pinecone", ping)


async def check_llm() -> str:
    async def ping():
        answer = await portkey_client.chat_completion([
            {"role": "system", "content": "This is a service health check. Reply only OK."},
            {"role": "user", "content": "Reply OK."},
        ])
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Gateway returned an empty response")
    return await _probe("llm", ping)


@router.get("", include_in_schema=False)
@router.get("/")
async def health_check(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    db_status, pinecone_status, llm_status = await asyncio.gather(
        check_db(db), check_pinecone(getattr(request.app.state, "pinecone_index", None)), check_llm(),
    )
    healthy = all(status == "ok" for status in (db_status, pinecone_status, llm_status))
    return JSONResponse(status_code=200 if healthy else 503, content={
        "status": "healthy" if healthy else "degraded", "db": db_status,
        "pinecone": pinecone_status, "llm": llm_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, headers={"Cache-Control": "no-store"})
