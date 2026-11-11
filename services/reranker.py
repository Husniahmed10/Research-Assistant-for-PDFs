"""
services/reranker.py — Cohere Reranker Service
===============================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import cohere client and flashrank (local fallback)
Step 2:  Initialize Cohere client with API key from settings

Step 3:  Write rerank_chunks(query: str, chunks: list[dict], top_n: int) → list[dict]:
            --- Primary: Cohere API Reranker ---
            - Call cohere.rerank(
                  model="rerank-english-v3.0",
                  query=query,
                  documents=[chunk["text"] for chunk in chunks],
                  top_n=top_n
              )
            - Cohere returns chunks sorted by relevance score
            - Map back to original chunk dicts with relevance_score added
            - Log reranking time and score distribution with Logfire

            --- Fallback: FlashRank (local, no API needed) ---
            - If Cohere fails (quota/network), use FlashRank locally
            - FlashRank is a small cross-encoder model that runs on CPU
            - Same interface: query + documents → sorted list

Step 4:  Return top_n chunks with highest relevance scores
         These are what gets passed to the LLM as context

NOTE: Reranking happens AFTER Pinecone retrieval.
      Pinecone retrieves top 20 candidates → Reranker picks best 5.
      This dramatically improves precision vs. sending all 20 to LLM.
"""

