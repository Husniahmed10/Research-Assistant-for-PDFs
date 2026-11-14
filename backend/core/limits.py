"""Shared service ceilings; document chunks must satisfy both token and byte limits."""

EMBEDDING_MAX_INPUT_TOKENS = 8192
PINECONE_MAX_METADATA_BYTES = 40 * 1024
# Reserve space for paper/page/index metadata and JSON framing.
CHUNK_MAX_TEXT_JSON_BYTES = PINECONE_MAX_METADATA_BYTES - 4096
