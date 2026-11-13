"""Deterministic source-label mapping shared by agent output and chat history."""

import re
from typing import Any


def build_citations(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cited sources by actual PDF page, retaining all source/chunk IDs."""
    sources = {str(chunk.get("source_id", number)): chunk for number, chunk in enumerate(chunks, 1)}
    references = list(dict.fromkeys(re.findall(r"\[(\d+)\]", answer)))
    if any(reference not in sources for reference in references):
        raise ValueError("Answer contains an unknown source reference")
    pages = {}
    for reference in references:
        chunk = sources[reference]
        page = int(chunk["page_num"])
        if page not in pages:
            pages[page] = {
                "page": page, "page_num": page, "id": reference,
                "chunk_id": chunk["chunk_id"], "text": chunk["text"],
                "snippet": chunk["text"][:100], "source_ids": [], "chunk_ids": [],
            }
        pages[page]["source_ids"].append(reference)
        pages[page]["chunk_ids"].append(chunk["chunk_id"])
    return list(pages.values())
