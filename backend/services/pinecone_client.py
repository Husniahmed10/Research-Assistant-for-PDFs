"""
services/pinecone_client.py — Pinecone Vector DB Client (Hybrid Search)
========================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------

--- Part A: Setup ---
Step 1:  Import Pinecone client and BM25Encoder from pinecone_text
Step 2:  Initialize Pinecone with API key from settings
Step 3:  Write init_pinecone_index() → Index:
            - Check if index with PINECONE_INDEX_NAME exists
            - If not, create it with:
                * dimension = 1536 (text-embedding-3-small)
                * metric = "dotproduct" (required for hybrid search)
                * pod_type = "p1" (or serverless spec)
            - Return the index object

--- Part B: BM25 Sparse Encoder ---
Step 4:  Initialize BM25Encoder (from pinecone_text):
            - Fit on the full document corpus during indexing
            - Saves sparse vectors alongside dense embeddings in Pinecone

--- Part C: Indexing ---
Step 5:  Write upsert_chunks(index, chunks_with_embeddings: list[dict], paper_id: str):
            - For each chunk:
                * dense_vector  = chunk["embedding"]
                * sparse_vector = bm25_encoder.encode_documents([chunk["text"]])[0]
                * metadata = { paper_id, chunk_index, page_num, text, parent_chunk_id }
            - Batch upsert to Pinecone (100 at a time)
            - Log with Logfire

--- Part D: Hybrid Search ---
Step 6:  Write hybrid_search(index, query: str, dense_vector: list[float], paper_id: str, top_k: int) → list[dict]:
            - Encode query as sparse vector: bm25_encoder.encode_queries([query])[0]
            - Run index.query() with BOTH dense and sparse vectors:
                * vector = dense_vector  (dense)
                * sparse_vector = sparse_q  (BM25)
                * alpha = settings.HYBRID_ALPHA  (0.7 = 70% dense, 30% sparse)
                * filter = {"paper_id": paper_id}  (only search within uploaded paper)
                * top_k = settings.TOP_K_RETRIEVAL
            - Return list of matches with text, score, and metadata

--- Part E: Cleanup ---
Step 7:  Write delete_paper_vectors(index, paper_id: str):
            - Delete all vectors where metadata.paper_id == paper_id
            - Called when a paper is deleted by user
"""

