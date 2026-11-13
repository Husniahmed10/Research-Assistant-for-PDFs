"""Studio entry point; exports the same workflow with JSON-safe runtime inputs.

Importing this module does not connect to Pinecone or read/write chat records.
Studio runs have their own checkpointed history and do not save API messages.
"""

from functools import lru_cache
from pathlib import Path
from threading import Lock
from uuid import UUID
from typing import Any
from typing_extensions import Required, TypedDict

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env", override=False)

from core.config import settings

# Studio can be started from either project root or backend/. Anchor its paths
# before importing database.py, which constructs the SQLAlchemy engine.
if not Path(settings.UPLOAD_DIR).is_absolute():
    settings.UPLOAD_DIR = str(BACKEND_DIR / settings.UPLOAD_DIR)
if not Path(settings.FLASHRANK_CACHE_DIR).is_absolute():
    settings.FLASHRANK_CACHE_DIR = str(BACKEND_DIR / settings.FLASHRANK_CACHE_DIR)
_database_url = make_url(settings.DATABASE_URL)
if (_database_url.get_backend_name() == "sqlite" and _database_url.database
        and _database_url.database != ":memory:" and not Path(_database_url.database).is_absolute()):
    settings.DATABASE_URL = _database_url.set(
        database=str(BACKEND_DIR / _database_url.database)
    ).render_as_string(hide_password=False)

from tracing.langsmith_setup import setup_langsmith

setup_langsmith()

import logfire
from langchain_core.runnables import RunnableConfig
from starlette.concurrency import run_in_threadpool

from agents.graph import build_graph
from agents.state import AgentState
from agents.nodes.input_guard import input_guard_node
from agents.nodes.retriever import retriever_node
from agents.nodes.citation_formatter import citation_formatter_node
from db import crud
from db.database import AsyncSessionLocal
from services import pinecone_client
from services.conversation_memory import extract_user_name

_index = None
_index_lock = Lock()


class StudioInput(TypedDict, total=False):
    query: Required[str]
    paper_id: str
    session_id: str
    chat_history: list[dict[str, Any]]
    max_generation_attempts: int


@lru_cache(maxsize=1)
def _configure_logfire() -> None:
    token = settings.LOGFIRE_TOKEN.strip()
    logfire.configure(token=token or None, send_to_logfire=bool(token), console=False,
                      service_name=settings.APP_NAME + " Studio")


def _get_index():
    global _index
    with _index_lock:
        if _index is None:
            index = pinecone_client.init_pinecone_index()
            pinecone_client.load_bm25_encoders()
            _index = index
        return _index


def _remember(state: AgentState, changes: AgentState) -> AgentState:
    messages = [*state.get("chat_history", []),
                {"role": "user", "content": state["query"]},
                {"role": "assistant", "content": changes.get("final_answer", state.get("final_answer", ""))}]
    name = extract_user_name([message["content"] for message in messages if message["role"] == "user"])
    return {**changes, "chat_history": messages[-4:], "user_name": name or state.get("user_name")}


async def _load_context(state: AgentState) -> AgentState:
    paper_id, session_id = state.get("paper_id"), state.get("session_id")
    paper_id = str(UUID(paper_id)) if paper_id else ""
    session_id = str(UUID(session_id)) if session_id else ""
    if paper_id or session_id:
        async with AsyncSessionLocal() as db:
            if session_id:
                session = await crud.get_session(db, session_id)
                if session is None:
                    raise ValueError("API session not found. Create one through FastAPI first.")
                if paper_id and paper_id != session.paper_id:
                    raise ValueError("The session belongs to a different paper.")
                paper_id = session.paper_id
            paper = await crud.get_paper(db, paper_id)
            if paper is None or paper.status != "ready":
                raise ValueError("Select a ready paper uploaded through the FastAPI papers API.")
            key = f"{paper_id}:{session_id}"
            if state.get("studio_context_key") != key:
                state["chat_history"] = []
                state["user_name"] = None
                if session_id:
                    messages = await crud.get_messages_for_session(db, session_id, limit=4)
                    state["chat_history"] = [{"role": m.role, "content": m.content} for m in messages]
                    state["user_name"] = extract_user_name(await crud.get_user_message_texts(db, session_id))
            state["studio_context_key"] = key
    elif state.get("studio_context_key"):
        state.update(chat_history=[], user_name=None, studio_context_key="")
    return {**state, "paper_id": paper_id, "session_id": session_id}


async def _studio_input(state: AgentState) -> AgentState:
    _configure_logfire()
    # Server checkpoints retain state between runs. Clear per-turn artifacts so a
    # prior failed/retried run cannot poison the next question or its sources.
    fresh: AgentState = {**state, "steps": [], "error": "", "query_type": "",
        "rewritten_query": "", "multi_queries": [], "hyde_document": "", "hyde_embedding": [],
        "retrieved_chunks": [], "reranked_chunks": [], "source_chunks": [], "citations": [],
        "final_answer": "", "model_used": "none", "is_safe": False, "is_output_safe": False,
        "rejection_reason": "", "output_rejection_reason": "", "generation_attempts": 0, "retry_count": 0}
    fresh = await _load_context(fresh)
    changes = await input_guard_node(fresh)
    if changes.get("query_type") == "conversation" or (not changes.get("is_safe") and not changes.get("error")):
        changes = _remember(fresh, changes)
    elif changes.get("is_safe") and not fresh.get("paper_id"):
        changes.update(is_safe=False, error="Provide a ready paper_id or an existing API session_id.",
                       final_answer="Choose a ready paper to run retrieval.")
    return {**fresh, **changes}


async def _studio_retriever(state: AgentState, config: RunnableConfig) -> AgentState:
    index = await run_in_threadpool(_get_index)
    # Inject the Python SDK object only on the server; it is never part of the
    # request, graph state, schema or checkpoint payload sent to Studio.
    config = {**config, "configurable": {**config.get("configurable", {}), "pinecone_index": index}}
    return await retriever_node(state, config)


async def _studio_citations(state: AgentState) -> AgentState:
    result = await citation_formatter_node(state)
    return _remember(state, result) if not result.get("error") and state.get("is_output_safe") else result


studio_agent = build_graph(input_node=_studio_input, retrieval_node=_studio_retriever,
                           citation_node=_studio_citations, input_schema=StudioInput).compile()
