"""
services/embedder.py — Text Embedding Service
===============================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import OpenAI client and settings
Step 2:  Initialize OpenAI client with API key from settings

Step 3:  Write embed_text(text: str) → list[float]:
            - Call openai.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=text)
            - Return the embedding vector (list of 1536 floats for text-embedding-3-small)
            - Wrap in Logfire span for tracing

Step 4:  Write embed_batch(texts: list[str]) → list[list[float]]:
            - Batch embedding for multiple texts (more efficient than calling embed_text in a loop)
            - OpenAI allows up to 2048 inputs per batch call
            - Split into batches of 100 if needed
            - Return list of embedding vectors

Step 5:  Write embed_chunks(chunks: list[dict]) → list[dict]:
            - For each chunk dict, add an "embedding" key with its vector
            - Uses embed_batch internally for efficiency
            - Log total tokens used and time taken

NOTE: These embeddings are dense vectors. 
      Sparse (BM25) vectors are computed separately in pinecone_client.py
"""

