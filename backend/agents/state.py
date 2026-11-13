"""Partial state updates shared by the research-paper agent nodes."""

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    paper_id: str
    session_id: str
    chat_history: list[dict[str, Any]]
    user_name: str | None  # User-supplied display name, scoped to this SQL session.
    studio_context_key: str  # Studio-only scope for checkpointed conversation context.
    is_safe: bool
    rejection_reason: str
    query_type: str
    rewritten_query: str
    multi_queries: list[str]
    hyde_document: str
    hyde_embedding: list[float]
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    source_chunks: list[dict[str, Any]]  # Exact, bounded excerpts sent to the LLM.
    final_answer: str
    citations: list[dict[str, Any]]
    model_used: str  # Actual response model, including gateway fallback models.
    is_output_safe: bool
    output_rejection_reason: str
    generation_attempts: int
    retry_count: int
    max_generation_attempts: int
    # Each node returns only its NEW progress messages; LangGraph accumulates them.
    steps: Annotated[list[str], operator.add]
    error: str
