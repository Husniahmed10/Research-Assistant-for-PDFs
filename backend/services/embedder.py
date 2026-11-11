"""Async embeddings with bounded batches and stable input/output ordering."""

from typing import Any

import logfire
from openai import AsyncOpenAI

from core.config import settings
from utils.text_utils import count_tokens


client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0, max_retries=2)
_MAX_BATCH_SIZE = 100
_MAX_INPUT_TOKENS = 8192
_MAX_BATCH_TOKENS = 300_000


async def embed_text(text: str) -> list[float]:
    """Return the configured embedding model's vector for one non-empty text."""
    return (await embed_batch([text]))[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed inputs in batches of at most 100, preserving their original order.

    Empty lists make no API calls. Validate all inputs before sending any batch;
    oversized sections raise an error instead of being silently truncated.
    Provider exceptions propagate after the SDK's bounded retries.
    """
    if not texts:
        return []
    token_counts = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Embedding input {index} must be a non-empty string")
        size = count_tokens(text, settings.OPENAI_EMBEDDING_MODEL)
        if size > _MAX_INPUT_TOKENS:
            raise ValueError(
                f"Embedding input {index} exceeds {_MAX_INPUT_TOKENS} tokens; "
                "split the section into smaller chunks"
            )
        token_counts.append(size)
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be configured to generate embeddings")

    vectors = []
    start = 0
    while start < len(texts):
        end = start
        total_tokens = 0
        while end < len(texts) and end - start < _MAX_BATCH_SIZE:
            if total_tokens + token_counts[end] > _MAX_BATCH_TOKENS:
                break
            total_tokens += token_counts[end]
            end += 1
        options: dict[str, Any] = {}
        if settings.OPENAI_EMBEDDING_MODEL.startswith("text-embedding-3-"):
            options["dimensions"] = settings.PINECONE_DIMENSION
        with logfire.span("embedding.batch", model=settings.OPENAI_EMBEDDING_MODEL,
                          input_count=end - start, estimated_tokens=total_tokens) as span:
            response = await client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=texts[start:end],
                encoding_format="float",
                **options,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                span.set_attribute("total_tokens", usage.total_tokens)
            logfire.info("Embedding batch completed", model=settings.OPENAI_EMBEDDING_MODEL,
                         input_count=end - start, estimated_tokens=total_tokens,
                         total_tokens=getattr(usage, "total_tokens", total_tokens))
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(end - start)):
            raise RuntimeError("Embedding response does not match the input batch")
        vectors.extend(item.embedding for item in ordered)
        start = end
    return vectors


async def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add an embedding to each chunk in place after all requests succeed."""
    vectors = await embed_batch([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks
