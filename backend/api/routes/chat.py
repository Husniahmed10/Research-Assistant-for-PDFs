"""Paper-scoped chat sessions, grounded answers, and persisted history."""

import asyncio
import json
import re
from contextlib import suppress
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any
from uuid import UUID

import logfire
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pinecone import Index
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from agents.graph import agent
from agents.state import AgentState
from core.config import settings
from core.dependencies import get_db, get_pinecone_index
from core.exceptions import ErrorHandlingRoute
from db import crud
from db.database import AsyncSessionLocal
from db.models import ChatSession, MessageEvaluation
from services import evaluator
from services.conversation_memory import extract_user_name, is_name_recall_query
from services.citations import build_citations
from utils.text_utils import count_tokens


router = APIRouter(prefix="/api/v1/chat", tags=["Chat"], route_class=ErrorHandlingRoute)


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
    page: int
    snippet: str
    source_ids: list[str]
    chunk_ids: list[str]


class AnswerResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    answer: str
    citations: list[Citation]
    model_used: str
    latency_ms: int
    steps: list[str]
    message_id: str | None = None


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
    try:
        return [Citation.model_validate(citation) for citation in build_citations(answer, chunks)]
    except (ValueError, KeyError, TypeError):
        raise HTTPException(status_code=502, detail="Answer contains an invalid source reference; please retry")


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


async def run_ragas_background(message_id: str, query: str, answer: str, contexts: list[str]) -> None:
    """Evaluate after response delivery; the worker owns its database session."""
    await evaluator.evaluate_message(message_id, query, answer, contexts)


async def _prepare_state(db: AsyncSession, session_id: str, query: str) -> AgentState:
    session = await _require_session(db, session_id)
    paper_id = session.paper_id
    await _require_ready_paper(db, paper_id)
    messages = await crud.get_messages_for_session(db, session_id, limit=4)
    history = [{"role": message.role, "content": message.content} for message in messages]
    user_name = None
    if is_name_recall_query(query):
        user_name = extract_user_name(await crud.get_user_message_texts(db, session_id))
    # Release the read transaction while external model/search calls run.
    await db.rollback()
    initial_state: AgentState = {
        "query": query, "paper_id": paper_id, "session_id": session_id,
        "chat_history": history, "steps": [], "error": "", "citations": [],
        "user_name": user_name,
        "source_chunks": [], "model_used": "none", "generation_attempts": 0,
        "retry_count": 0, "max_generation_attempts": 2,
        "is_safe": False, "is_output_safe": False, "final_answer": "",
    }
    return initial_state


def _answer_response(result: AgentState, started: float) -> AnswerResponse:
    if result.get("error"):
        if result.get("output_rejection_reason"):
            raise HTTPException(status_code=502, detail="Could not verify an answer from the paper; please retry")
        raise HTTPException(status_code=503, detail="Could not generate an answer; please retry")
    if result.get("is_safe") is True and result.get("is_output_safe") is not True:
        raise HTTPException(status_code=502, detail="Answer could not be verified")
    answer = result["final_answer"]
    model_used = result.get("model_used", "none")
    citations = [Citation.model_validate(citation) for citation in result["citations"]]
    latency_ms = max(0, round((perf_counter() - started) * 1000))
    response = AnswerResponse(
        answer=answer, citations=citations, model_used=model_used, latency_ms=latency_ms,
        steps=result["steps"],
    )
    return response


async def _persist_answer(db: AsyncSession, state: AgentState, result: AgentState,
                          started: float, background_tasks: BackgroundTasks) -> AnswerResponse:
    response = _answer_response(result, started)
    answer, model_used, latency_ms = response.answer, response.model_used, response.latency_ms
    chunks = result.get("source_chunks", [])
    try:
        async with db.begin():
            # The paper/session could have been deleted during generation.
            await _require_session(db, state["session_id"])
            await _require_ready_paper(db, state["paper_id"])
            user = await crud.add_message(db, state["session_id"], "user", state["query"])
            assistant = await crud.add_message(
                db, state["session_id"], "assistant", answer, retrieved_chunks=chunks,
                model_used=model_used, latency_ms=latency_ms,
            )
            if assistant.created_at <= user.created_at:
                assistant.created_at = user.created_at + timedelta(microseconds=1)
            message_id = assistant.id
            should_evaluate = result.get("is_output_safe") is True and bool(chunks) and model_used != "none"
            db.add(MessageEvaluation(message_id=message_id, status="pending" if should_evaluate else "skipped"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Could not save the conversation; please retry") from exc
    response.message_id = message_id
    if should_evaluate:
        background_tasks.add_task(run_ragas_background, message_id=message_id, query=state["query"],
                                  answer=answer, contexts=[chunk["text"] for chunk in chunks])
    return response


@router.post("/sessions/{session_id}/message", response_model=AnswerResponse)
async def send_message(
    session_id: UUID,
    body: MessageRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    index: Index = Depends(get_pinecone_index),
) -> AnswerResponse:
    started = perf_counter()
    state = await _prepare_state(db, str(session_id), body.query)
    try:
        result = await agent.ainvoke(state, config={
            "configurable": {"pinecone_index": index}, "recursion_limit": 40,
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not generate an answer; please retry") from exc
    return await _persist_answer(db, state, result, started, background_tasks)


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n\n"


async def _stream_answer(state: AgentState, index: Index, background_tasks: BackgroundTasks,
                         started: float):
    """Expose progress immediately, but release only a verified, saved answer."""
    result = {**state, "steps": list(state.get("steps", []))}
    updates = agent.astream(state, config={
        "configurable": {"pinecone_index": index}, "recursion_limit": 40,
    }, stream_mode="updates")
    pending = None
    try:
        yield _sse({"type": "step", "content": "Checking input safety..."})
        pending = asyncio.create_task(anext(updates))
        while True:
            ready, _ = await asyncio.wait({pending}, timeout=15)
            if not ready:
                yield ": keep-alive\n\n"
                continue
            try:
                changes = pending.result()
            except StopAsyncIteration:
                break
            for node, node_update in changes.items():
                if not node_update:
                    continue
                for key, value in node_update.items():
                    if key == "steps":
                        result["steps"] += value
                        for step in value:
                            yield _sse({"type": "step", "content": step})
                    else:
                        result[key] = value
                next_step = {
                    "hyde_generator": "Running hybrid search...",
                    "retriever": "Reranking results...",
                    "reranker": "Generating answer...",
                }.get(node)
                if next_step and not result.get("error"):
                    yield _sse({"type": "step", "content": next_step})
            pending = asyncio.create_task(anext(updates))
        # Request dependencies may already be closed in the pinned FastAPI.
        async with AsyncSessionLocal() as db:
            response = await _persist_answer(db, state, result, started, background_tasks)
        # These are whitespace-preserving pieces of the validated final answer,
        # not an unchecked provider draft (which could be rejected or retried).
        for piece in re.findall(r"\s*\S+|\s+$", response.answer):
            yield _sse({"type": "token", "content": piece})
        yield _sse({"type": "done", "citations": [c.model_dump() for c in response.citations],
                    "model_used": response.model_used, "message_id": response.message_id,
                    "latency_ms": response.latency_ms})
    except HTTPException as exc:
        if exc.status_code >= 500:
            logfire.error("Chat stream failed", error_type=type(exc).__name__, status_code=exc.status_code)
        yield _sse({"type": "error", "content": exc.detail, "status_code": exc.status_code})
    except Exception as exc:
        logfire.error("Chat stream failed", error_type=type(exc).__name__)
        yield _sse({"type": "error", "content": "Could not complete the answer; please retry", "status_code": 503})
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                await pending
        await updates.aclose()


@router.get("/sessions/{session_id}/stream", response_class=StreamingResponse)
async def stream_message(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    query: str = Query(..., min_length=1, max_length=8000),
    db: AsyncSession = Depends(get_db),
    index: Index = Depends(get_pinecone_index),
) -> StreamingResponse:
    started = perf_counter()
    # Reuse POST validation, including the token budget, before sending headers.
    try:
        body = MessageRequest(query=query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Question must be non-empty and at most 2000 tokens") from exc
    state = await _prepare_state(db, str(session_id), body.query)
    return StreamingResponse(_stream_answer(state, index, background_tasks, started),
                             media_type="text/event-stream", background=background_tasks,
                             headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})


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
