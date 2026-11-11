"""Async database operations.

Writes flush changes but do not commit. Callers must await db.commit() to persist
them, or use async with db.begin() to group several operations atomically.
File and vector cleanup belong in the service layer, not database CRUD.
"""

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ChatSession, Message, Paper


async def create_paper(
    db: AsyncSession,
    filename: str,
    file_path: str,
    file_size_kb: float,
    page_count: int | None = None,
    full_text: str | None = None,
    *,
    paper_id: str | None = None,
    status: str = "uploading",
) -> Paper:
    """Insert a paper, optionally using an upload's pre-generated ID/status."""
    if status not in {"uploading", "processing", "ready", "failed"}:
        raise ValueError("Unsupported paper status")
    paper = Paper(
        id=paper_id,
        filename=filename,
        file_path=file_path,
        file_size_kb=file_size_kb,
        page_count=page_count,
        full_text=full_text,
        status=status,
    )
    db.add(paper)
    await db.flush()
    return paper


async def get_paper(db: AsyncSession, paper_id: str) -> Paper | None:
    """Return a paper by ID, or None when it does not exist."""
    return await db.get(Paper, paper_id)


async def get_all_papers(db: AsyncSession) -> list[Paper]:
    """List papers newest first with a stable tie-breaker."""
    result = await db.scalars(
        select(Paper).order_by(Paper.created_at.desc(), Paper.id.desc())
    )
    return list(result.all())


async def update_paper_status(
    db: AsyncSession,
    paper_id: str,
    status: str,
    page_count: int | None = None,
) -> Paper | None:
    """Change processing status and optionally set the extracted page count."""
    if status not in {"uploading", "processing", "ready", "failed"}:
        raise ValueError("Unsupported paper status")
    paper = await get_paper(db, paper_id)
    if paper is None:
        return None
    paper.status = status
    if page_count is not None:
        paper.page_count = page_count
    await db.flush()
    return paper


async def update_paper_summary(
    db: AsyncSession, paper_id: str, summary: dict[str, Any] | str
) -> Paper | None:
    """Store a summary object or JSON object string in the Text column."""
    summary_data = json.loads(summary) if isinstance(summary, str) else summary
    if not isinstance(summary_data, dict):
        raise ValueError("Summary must be a JSON object")
    serialized = json.dumps(summary_data, ensure_ascii=False, allow_nan=False)
    paper = await get_paper(db, paper_id)
    if paper is None:
        return None
    paper.summary_json = serialized
    await db.flush()
    return paper


async def delete_paper(db: AsyncSession, paper_id: str) -> bool:
    """Delete a paper and its related sessions/messages; return whether found."""
    paper = await get_paper(db, paper_id)
    if paper is None:
        return False
    await db.delete(paper)
    await db.flush()
    return True


async def create_session(
    db: AsyncSession, paper_id: str, title: str | None = None
) -> ChatSession:
    """Insert a conversation; reject a missing parent paper."""
    paper = await get_paper(db, paper_id)
    if paper is None:
        raise ValueError("Paper not found")
    session = ChatSession(paper=paper, title=title)
    db.add(session)
    await db.flush()
    return session


async def get_sessions_for_paper(
    db: AsyncSession, paper_id: str
) -> list[ChatSession]:
    """List a paper's conversations newest first."""
    result = await db.scalars(
        select(ChatSession)
        .where(ChatSession.paper_id == paper_id)
        .order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
    )
    return list(result.all())


async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    model_used: str | None = None,
    latency_ms: int | None = None,
) -> Message:
    """Insert a user/assistant message in an existing conversation."""
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be user or assistant")
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise ValueError("Chat session not found")
    message = Message(
        session=session,
        role=role,
        content=content,
        retrieved_chunks=retrieved_chunks if retrieved_chunks is not None else [],
        model_used=model_used,
        latency_ms=latency_ms,
    )
    db.add(message)
    await db.flush()
    return message


async def get_messages_for_session(
    db: AsyncSession, session_id: str
) -> list[Message]:
    """Return chat history oldest first with a stable tie-breaker."""
    result = await db.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    return list(result.all())
