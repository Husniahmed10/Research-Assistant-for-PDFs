"""Node 7: rerank fused candidates using Cohere or local FlashRank."""

from starlette.concurrency import run_in_threadpool

from agents.state import AgentState
from agents.nodes._shared import UNVERIFIED, question, skipped, update
from core.config import settings
from services import reranker


async def reranker_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return update("No retrieved chunks to rerank", reranked_chunks=[])
    try:
        ranked = await run_in_threadpool(
            reranker.rerank_chunks, question(state), chunks, settings.TOP_K_RERANK
        )
        return update(f"Reranked to {len(ranked)} chunks", reranked_chunks=ranked)
    except Exception:
        return update("Reranking failed", reranked_chunks=[], error="Reranking unavailable",
                      final_answer=UNVERIFIED, citations=[], is_output_safe=False)
