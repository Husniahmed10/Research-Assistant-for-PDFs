"""Paper-scoped chat sessions, grounded answers, and persisted history."""

import re
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pinecone import Index
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.dependencies import get_db, get_pinecone_index
from db import crud
from db.models import ChatSession
from services.simple_rag import simple_rag
from utils.text_utils import count_tokens


router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


class CreateSessionRequest(BaseModel):
    paper_id: UUID


class SessionResponse(BaseModel):
    session_id: str
    paper_id: str
    created_at: datetime


class MessageRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        if count_tokens(value, settings.OPENAI_MODEL.rsplit("/", 1)[-1]) > 2000:
            raise ValueError("Question exceeds the 2000-token limit")
        return value


class Citation(BaseModel):
    id: str
    chunk_id: str
    page_num: int
    text: str


class AnswerResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    answer: str
    citations: list[Citation]
    model_used: str
    latency_ms: int


class MessageResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    id: str
    role: str
    content: str
    citations: list[Citation]
    model_used: str | None
    latency_ms: int | None
    created_at: datetime


def _citations(answer: str, chunks: list[dict[str, Any]]) -> list[Citation]:
    """Map model source labels to retrieved evidence; never invent page numbers."""
    sources = {str(chunk.get("source_id", number)): chunk for number, chunk in enumerate(chunks, 1)}
    references = list(dict.fromkeys(re.findall(r"\[(\d+)\]", answer)))
    if any(reference not in sources for reference in references):
        raise HTTPException(status_code=502, detail="Answer contains an invalid source reference; please retry")
    return [Citation(
        id=reference,
        chunk_id=sources[reference]["chunk_id"],
        page_num=sources[reference]["page_num"],
        text=sources[reference]["text"],
    ) for reference in references]


async def _require_session(db: AsyncSession, session_id: str) -> ChatSession:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


async def _require_ready_paper(db: AsyncSession, paper_id: str) -> None:
    paper = await crud.get_paper(db, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    if paper.status != "ready":
        raise HTTPException(status_code=409, detail="Paper is not ready for chat")


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_chat_session(
    body: CreateSessionRequest, db: AsyncSession = Depends(get_db)
) -> SessionResponse:
    paper_id = str(body.paper_id)
    await _require_ready_paper(db, paper_id)
    session = await crud.create_session(db, paper_id, title="New Chat")
    response = SessionResponse(
        session_id=session.id, paper_id=session.paper_id, created_at=session.created_at
    )
    await db.commit()
    return response


@router.post("/sessions/{session_id}/message", response_model=AnswerResponse)
async def send_message(
    session_id: UUID,
    body: MessageRequest,
    db: AsyncSession = Depends(get_db),
    index: Index = Depends(get_pinecone_index),
) -> AnswerResponse:
    started = perf_counter()
    session_id = str(session_id)
    session = await _require_session(db, session_id)
    paper_id = session.paper_id
    await _require_ready_paper(db, paper_id)
    messages = await crud.get_messages_for_session(db, session_id, limit=4)
    history = [{"role": message.role, "content": message.content} for message in messages]
    # Release the read transaction while external model/search calls run.
    await db.rollback()
    try:
        result = await simple_rag(body.query, paper_id, history, index)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not generate an answer; please retry") from exc
    answer = result["answer"]
    chunks = result["chunks_used"]
    model_used = result.get("model_used", "unknown")
    citations = _citations(answer, chunks)
    latency_ms = max(0, round((perf_counter() - started) * 1000))
    response = AnswerResponse(
        answer=answer, citations=citations, model_used=model_used, latency_ms=latency_ms
    )
    try:
        async with db.begin():
            # The paper/session could have been deleted during generation.
            await _require_session(db, session_id)
            await _require_ready_paper(db, paper_id)
            user = await crud.add_message(db, session_id, "user", body.query)
            assistant = await crud.add_message(
                db, session_id, "assistant", answer, retrieved_chunks=chunks,
                model_used=model_used, latency_ms=latency_ms,
            )
            if assistant.created_at <= user.created_at:
                assistant.created_at = user.created_at + timedelta(microseconds=1)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Could not save the conversation; please retry") from exc
    return response


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_chat_history(
    session_id: UUID, db: AsyncSession = Depends(get_db)
) -> list[MessageResponse]:
    await _require_session(db, str(session_id))
    messages = await crud.get_messages_for_session(db, str(session_id))
    return [MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        citations=_citations(message.content, message.retrieved_chunks) if message.role == "assistant" else [],
        model_used=message.model_used,
        latency_ms=message.latency_ms,
        created_at=message.created_at,
    ) for message in messages]
