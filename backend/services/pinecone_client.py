"""Dense/sparse hybrid vector storage using the pinned Pinecone 4.x SDK.

All functions are synchronous: call them through a thread pool from async routes.
Operations share the default namespace and always scope reads/deletes by paper_id.
Importing this module does not connect to Pinecone or create an index.
"""

import json
import math
import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

import logfire
from pinecone import Index, Pinecone, ServerlessSpec
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder

from core.config import settings


_BATCH_SIZE = 100
# Leave room for request framing beneath the service's 2 MB upsert limit.
_MAX_BATCH_BYTES = 1_900_000
_MAX_METADATA_BYTES = 40 * 1024


def _bm25_path(paper_id: str | None = None) -> Path:
    root = Path(settings.UPLOAD_DIR).resolve()
    if paper_id is None:
        return root / "bm25_params.json"
    _require_paper_id(paper_id)
    # IDs cannot influence filesystem paths, even for direct service callers.
    return root / "bm25" / sha256(paper_id.encode("utf-8")).hexdigest() / "bm25_params.json"


def fit_bm25(corpus: list[str], paper_id: str | None = None) -> BM25Encoder:
    """Fit all chunk texts and atomically save bm25_params.json.

    Pass paper_id during indexing to isolate corpus statistics for each paper.
    Without an ID, save a standalone encoder under UPLOAD_DIR/bm25_params.json.
    """
    if not corpus or any(not isinstance(text, str) or not text.strip() for text in corpus):
        raise ValueError("BM25 corpus must contain non-empty strings")
    encoder = BM25Encoder()
    try:
        encoder.fit(corpus)
    except ZeroDivisionError as exc:
        raise ValueError("Corpus contains no searchable BM25 terms") from exc
    path = _bm25_path(paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        encoder.dump(str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return encoder


@lru_cache(maxsize=128)
def _load_bm25(path: str, modified_ns: int) -> BM25Encoder:
    """Load immutable fitted parameters; mtime invalidates replaced files."""
    return BM25Encoder().load(path)


def load_bm25_encoders() -> int:
    """Load saved standalone/per-paper encoders at startup; return their count."""
    root = Path(settings.UPLOAD_DIR).resolve()
    paths = list((root / "bm25").glob("*/bm25_params.json"))
    if (root / "bm25_params.json").is_file():
        paths.append(root / "bm25_params.json")
    for path in paths:
        _load_bm25(str(path), path.stat().st_mtime_ns)
    return len(paths)


def _require_paper_id(paper_id: str) -> None:
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("paper_id must be a non-empty string")


def _validate_vector(vector: list[float]) -> None:
    if len(vector) != settings.PINECONE_DIMENSION:
        raise ValueError(f"Embedding must have {settings.PINECONE_DIMENSION} dimensions")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in vector):
        raise ValueError("Embedding values must be finite numbers")


def init_pinecone_index() -> Index:
    """Open or create the configured dotproduct serverless index.

    PINECONE_ENVIRONMENT supplies the AWS region (default us-east-1). Dimension
    defaults to 1536 and must match the embedding service. Creation waits at most
    60 seconds for readiness; errors propagate so callers can retry/report them.
    """
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY must be configured")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    name = settings.PINECONE_INDEX_NAME
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=settings.PINECONE_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT),
            timeout=60,
        )
    description = pc.describe_index(name)
    if description.dimension != settings.PINECONE_DIMENSION:
        raise ValueError("Existing Pinecone index dimension does not match PINECONE_DIMENSION")
    if description.metric != "dotproduct":
        raise ValueError("Existing Pinecone index must use the dotproduct metric")
    if not description.status["ready"]:
        raise RuntimeError("Pinecone index is not ready; retry initialization shortly")
    return pc.Index(name)


def upsert_chunks(
    index: Index, chunks_with_embeddings: list[dict[str, Any]], paper_id: str
) -> int:
    """Validate and upsert chunks in batches of at most 100; return their count.

    Smaller batches are used if vector payloads approach the request byte limit.
    Validation completes before the first write. Network failures may still leave
    earlier batches stored; retry the same chunk IDs or delete this paper's vectors.
    """
    _require_paper_id(paper_id)
    vectors = []
    seen_ids = set()
    for chunk in chunks_with_embeddings:
        chunk_id = chunk["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen_ids:
            raise ValueError("Each chunk must have a unique non-empty chunk_id")
        seen_ids.add(chunk_id)
        values = chunk["embedding"]
        _validate_vector(values)
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError("Chunk text must be a non-empty string")
        if type(chunk["page_num"]) is not int or chunk["page_num"] < 1:
            raise ValueError("page_num must be a positive integer")
        if type(chunk["chunk_index"]) is not int or chunk["chunk_index"] < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        metadata = {
            "paper_id": paper_id,
            "text": chunk["text"],
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
        }
        if len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) > _MAX_METADATA_BYTES:
            raise ValueError("Chunk metadata exceeds 40 KB; split the section into smaller chunks")
        vector = {"id": chunk_id, "values": values, "metadata": metadata}
        vectors.append(vector)

    if not vectors:
        return 0
    encoder = fit_bm25([chunk["text"] for chunk in chunks_with_embeddings], paper_id)
    sparse_vectors = encoder.encode_documents([chunk["text"] for chunk in chunks_with_embeddings])
    vector_sizes = []
    for vector, sparse in zip(vectors, sparse_vectors):
        # Stopword/punctuation-only chunks can legitimately have no sparse terms.
        if sparse["indices"]:
            vector["sparse_values"] = sparse
        size = len(json.dumps(vector, allow_nan=False).encode("utf-8"))
        if size > _MAX_BATCH_BYTES:
            raise ValueError("A chunk exceeds the upsert request size limit")
        vector_sizes.append(size)

    batch = []
    batch_size_bytes = 0
    for vector, size in zip(vectors, vector_sizes):
        if batch and (len(batch) >= _BATCH_SIZE or batch_size_bytes + size > _MAX_BATCH_BYTES):
            index.upsert(vectors=batch)
            batch = []
            batch_size_bytes = 0
        batch.append(vector)
        batch_size_bytes += size
    if batch:
        index.upsert(vectors=batch)
    return len(vectors)


def dense_search(
    index: Index, query_embedding: list[float], paper_id: str, top_k: int
) -> list[dict[str, Any]]:
    """Return ranked text/page/score/chunk_id results for the selected paper."""
    _require_paper_id(paper_id)
    _validate_vector(query_embedding)
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return _query_index(index, paper_id, "dense",
        vector=query_embedding,
        filter={"paper_id": paper_id},
        top_k=top_k,
        include_metadata=True,
    )


def _query_index(index: Index, paper_id: str, search_type: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Record search metrics without logging query text, embeddings or passages."""
    with logfire.span("pinecone.search", search_type=search_type, top_k=kwargs["top_k"]) as span:
        started = perf_counter()
        try:
            results = _search_results(index.query(**kwargs), paper_id)
            span.set_attribute("result_count", len(results))
            logfire.info("Pinecone search completed", search_type=search_type,
                         latency_ms=(perf_counter() - started) * 1000, result_count=len(results))
            return results
        finally:
            span.set_attribute("latency_ms", (perf_counter() - started) * 1000)


def _search_results(response: Any, paper_id: str) -> list[dict[str, Any]]:
    results = []
    for match in response.get("matches", []) or []:
        metadata = match.get("metadata") or {}
        # Defense in depth: never return another paper's passages even if an
        # unexpected response ignores the query's filter.
        if metadata.get("paper_id") != paper_id:
            continue
        results.append({
            "text": metadata["text"],
            "page_num": int(metadata["page_num"]),
            "score": float(match["score"]),
            "chunk_id": match["id"],
        })
    return results


def hybrid_search(
    index: Index, query: str, dense_vector: list[float], paper_id: str, top_k: int
) -> list[dict[str, Any]]:
    """Query both signals with alpha-weighted vectors and a paper filter.

    Pinecone combines dense/sparse dotproducts into one ranked list. It has no
    alpha query argument: scaling happens here. Missing parameters identify a
    legacy dense-only paper; use dense retrieval until that paper is re-indexed.
    """
    _require_paper_id(paper_id)
    _validate_vector(dense_vector)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    alpha = settings.HYBRID_ALPHA
    if not 0 <= alpha <= 1:
        raise ValueError("HYBRID_ALPHA must be between 0 and 1")
    path = _bm25_path(paper_id)
    if alpha == 1 or not path.is_file():
        return dense_search(index, dense_vector, paper_id, top_k)
    encoder = _load_bm25(str(path), path.stat().st_mtime_ns)
    sparse = encoder.encode_queries([query])[0]
    if not sparse["indices"]:
        return dense_search(index, dense_vector, paper_id, top_k)
    dense, sparse = hybrid_convex_scale(dense_vector, sparse, alpha)
    return _query_index(index, paper_id, "hybrid",
        vector=dense, sparse_vector=sparse, filter={"paper_id": paper_id},
        top_k=top_k, include_metadata=True,
    )


def delete_paper_vectors(index: Index, paper_id: str) -> None:
    """Delete only vectors matching a non-empty paper ID; never delete_all."""
    _require_paper_id(paper_id)
    index.delete(filter={"paper_id": paper_id})
    path = _bm25_path(paper_id)
    path.unlink(missing_ok=True)
    _load_bm25.cache_clear()
