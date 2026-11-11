"""Node 6: paper-scoped hybrid retrieval and reciprocal rank fusion."""

import asyncio

from langchain_core.runnables import RunnableConfig
from starlette.concurrency import run_in_threadpool

from agents.state import AgentState
from agents.nodes._shared import UNVERIFIED, question, skipped, update
from core.config import settings
from services import embedder, pinecone_client
from utils.text_utils import apply_rrf


async def retriever_node(state: AgentState, config: RunnableConfig = None) -> AgentState:
    if skipped(state):
        return {}
    try:
        index = (config or {}).get("configurable", {}).get("pinecone_index")
        if index is None:
            raise ValueError("Pass the application index in config.configurable.pinecone_index")
        queries = list(dict.fromkeys(state.get("multi_queries") or [question(state)]))[:4]
        vectors = await embedder.embed_batch(queries)
        searches = [(query, vector) for query, vector in zip(queries, vectors)]
        if state.get("hyde_embedding"):
            # Keep sparse terms tied to the question, not fabricated HyDE terms.
            searches.append((question(state), state["hyde_embedding"]))
        ranked_lists = await asyncio.gather(*[
            run_in_threadpool(pinecone_client.hybrid_search, index, query, vector,
                              state["paper_id"], settings.TOP_K_RETRIEVAL)
            for query, vector in searches
        ])
        merged = apply_rrf(ranked_lists)[:settings.TOP_K_RETRIEVAL]
        return update(f"Retrieved {len(merged)} unique chunks from {len(searches)} searches",
                      retrieved_chunks=merged)
    except Exception:
        return update("Retrieval failed", retrieved_chunks=[], error="Paper retrieval unavailable",
                      final_answer=UNVERIFIED, citations=[], is_output_safe=False)
