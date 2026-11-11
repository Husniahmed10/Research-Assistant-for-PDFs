"""
agents/nodes/reranker.py — Cohere Reranker Node (Node 7)
=========================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import reranker service and AgentState

Step 2:  Write reranker_node(state: AgentState) → dict:
            - Extract state["retrieved_chunks"] and state["query"]
            - Call reranker.rerank_chunks(
                  query=state["rewritten_query"],
                  chunks=state["retrieved_chunks"],
                  top_n=settings.TOP_K_RERANK   # typically 5
              )
            - Set state["reranked_chunks"] = top 5 chunks
            - Append "🏆 Reranked to top {N} most relevant chunks" to state["steps"]
            - Return updated state dict

NOTE: This is the final retrieval step before LLM generation.
      The LLM will ONLY see these top-N chunks as context.
      This prevents "lost in the middle" problem (too much context dilutes attention).
"""

