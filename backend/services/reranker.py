"""Cohere reranking with a cached, local FlashRank cross-encoder fallback.

These helpers are synchronous. Use a thread pool from async application code.
"""

import logging
import math
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import cohere
import httpx
from flashrank import Ranker, RerankRequest

from core.config import settings


logger = logging.getLogger(__name__)
_local_lock = Lock()


@lru_cache(maxsize=2)
def _get_local_ranker(model: str, cache_dir: str) -> Ranker:
    return Ranker(model_name=model, cache_dir=cache_dir, log_level="WARNING")


def prepare_flashrank() -> None:
    """Cache/load local weights once so subsequent fallback calls work offline.

    The first call downloads weights if absent. Keep FLASHRANK_CACHE_DIR when
    deploying; failed downloads/model loads propagate and can be retried.
    """
    with _local_lock:
        _get_local_ranker(settings.FLASHRANK_MODEL, str(Path(settings.FLASHRANK_CACHE_DIR).resolve()))


def _map_results(
    scored_indices: list[tuple[int, float]], chunks: list[dict[str, Any]], top_n: int
) -> list[dict[str, Any]]:
    seen = set()
    ranked = []
    for index, score in scored_indices:
        if type(index) is not int or not 0 <= index < len(chunks) or index in seen:
            raise ValueError("Reranker returned an invalid or duplicate document index")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Reranker returned a non-finite relevance score")
        seen.add(index)
        ranked.append((index, score))
    if len(ranked) < top_n:
        raise ValueError("Reranker returned too few documents")
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [{**chunks[index], "relevance_score": score} for index, score in ranked[:top_n]]


def rerank_chunks(
    query: str, chunks: list[dict[str, Any]], top_n: int
) -> list[dict[str, Any]]:
    """Return top chunks sorted by relevance without mutating caller data.

    Try Cohere first when configured. Missing credentials, API errors or malformed
    results trigger FlashRank. If both fail, raise instead of inventing scores.
    Relevance scores are ranking signals, not probabilities of answer accuracy.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if type(top_n) is not int or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")
    if not chunks or top_n == 0:
        return []
    if any(not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str)
           or not chunk["text"].strip() for chunk in chunks):
        raise ValueError("Every chunk must contain non-empty text")
    top_n = min(top_n, len(chunks))
    api_key = settings.COHERE_API_KEY
    if api_key and "your-" not in api_key and "..." not in api_key:
        try:
            with httpx.Client(timeout=20.0) as http_client:
                co = cohere.Client(api_key=api_key, timeout=20.0, httpx_client=http_client)
                response = co.rerank(
                    model=settings.COHERE_RERANK_MODEL,
                    query=query,
                    documents=[chunk["text"] for chunk in chunks],
                    top_n=top_n,
                )
            return _map_results(
                [(result.index, result.relevance_score) for result in response.results],
                chunks, top_n,
            )
        except Exception:
            # Provider error payloads can include credentials or document text.
            logger.warning("Cohere reranking failed; trying local FlashRank")

    try:
        with _local_lock:
            ranker = _get_local_ranker(
                settings.FLASHRANK_MODEL, str(Path(settings.FLASHRANK_CACHE_DIR).resolve())
            )
            # FlashRank modifies and sorts these passage dictionaries in place.
            passages = [{"id": index, "text": chunk["text"]} for index, chunk in enumerate(chunks)]
            results = ranker.rerank(RerankRequest(query=query, passages=passages))
        return _map_results([(result["id"], result["score"]) for result in results], chunks, top_n)
    except Exception as exc:
        raise RuntimeError(
            "Local reranking failed; prepare the FlashRank cache and retry"
        ) from exc
