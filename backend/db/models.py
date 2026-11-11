"""Database models for uploaded papers and their conversations.

Timestamps are stored as naive UTC datetimes for SQLite compatibility.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from db.database import Base


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Paper(Base):
    """An uploaded PDF and its extracted content and summary."""

    __tablename__ = "papers"

    id = Column(String(36), primary_key=True, default=_new_id)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_kb = Column(Float, nullable=False)
    page_count = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="uploading")
    summary_json = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utc_now)

    sessions = relationship(
        "ChatSession",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ChatSession(Base):
    """A conversation associated with one paper."""

    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=_new_id)
    paper_id = Column(String(36), ForeignKey("papers.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utc_now)

    paper = relationship("Paper", back_populates="sessions")
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="(Message.created_at, Message.id)",
    )


class Message(Base):
    """A user or assistant message with optional retrieval metadata."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=_new_id)
    session_id = Column(
        String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON, nullable=False, default=list)
    model_used = Column(String(255), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utc_now)

    session = relationship("ChatSession", back_populates="messages")
