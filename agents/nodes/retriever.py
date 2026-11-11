"""
agents/nodes/retriever.py — Hybrid Search Retriever (Node 6)
=============================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import pinecone_client, embedder, and AgentState

Step 2:  Write retriever_node(state: AgentState) → dict:

            --- Strategy: Run multiple queries, merge results ---

            For each query in state["multi_queries"]:
                a) Embed the query using embedder.embed_text(query)
                b) Run hybrid_search() from pinecone_client:
                    - Uses BOTH dense (OpenAI embedding) AND sparse (BM25) vectors
                    - filter = { "paper_id": state["paper_id"] }
                    - top_k = settings.TOP_K_RETRIEVAL (e.g., 20)

            Also run a separate search using state["hyde_document"] embedding
            (the HyDE embedding often retrieves the most relevant chunks)

            --- Merge results ---
            - Collect all retrieved chunks from all queries
            - Deduplicate by chunk_id (a chunk retrieved by multiple queries
              likely gets a relevance boost)
            - Apply RRF (Reciprocal Rank Fusion) to merge ranked lists:
                RRF score = sum(1 / (rank + k)) for each query list
            - Sort by combined RRF score, take top TOP_K_RETRIEVAL unique chunks

            --- Parent chunk expansion ---
            - For each retrieved child chunk, fetch its parent chunk from DB
            - LLM will receive the larger parent context

            - Set state["retrieved_chunks"] = merged, deduplicated chunks
            - Append "🔍 Retrieved {N} chunks via hybrid search" to state["steps"]
            - Return updated state dict
"""

