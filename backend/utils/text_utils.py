"""Text normalization, token budgeting, and research-paper heading detection."""

import re
import math
import unicodedata
from typing import Any

import tiktoken


_SECTION_PATTERN = re.compile(
    r"^[ \t]*(?:(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?[ \t]+)?"
    r"(?P<header>Abstract|Introduction|Related[ \t]+Work|Methodology|Methods|"
    r"Experiments|Results(?:[ \t]+and[ \t]+Discussion)?|Discussion|Conclusions?|References)"
    r"[ \t]*[:.\-–—]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(text: str) -> str:
    """Normalize compatibility Unicode, remove nulls, and collapse whitespace.

    This flattens line breaks. Detect section headers on the original extracted
    text before applying this helper if line boundaries are needed.
    """
    text = unicodedata.normalize("NFKC", text.replace("\x00", ""))
    text = "".join(
        character for character in text
        if character.isspace() or unicodedata.category(character) != "Cc"
    )
    return " ".join(text.split())


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count ordinary text tokens using the named model's encoding.

    Treat special-token-looking strings in uploaded text as ordinary content.
    Unknown model names raise tiktoken's KeyError rather than guessing an encoding.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text, disallowed_special=()))


def truncate_to_tokens(
    text: str, max_tokens: int, model: str = "gpt-4o"
) -> str:
    """Return a token-limited prefix without introducing broken Unicode."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    # A token boundary can split a UTF-8 character. Drop only an incomplete
    # trailing character instead of introducing the replacement character.
    truncated = encoding.decode_bytes(tokens[:max_tokens]).decode("utf-8", errors="ignore")
    # Retokenization of the shortened Unicode prefix must still fit the budget.
    while len(encoding.encode(truncated, disallowed_special=())) > max_tokens:
        truncated = truncated[:-1]
    return truncated


def extract_section_headers(text: str) -> list[str]:
    """Return heading names in document order, without section numbering.

    Matches standalone heading lines, not occurrences inside prose. Supports
    numbered headings, Roman numerals, case variants, and common related sections.
    """
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return [" ".join(match.group("header").split()).title()
            for match in _SECTION_PATTERN.finditer(text)]


def apply_rrf(
    ranked_lists: list[list[str | dict[str, Any]]], k: float = 60
) -> list[str | dict[str, Any]]:
    """Fuse ranked document IDs or chunks using 1 / (k + one-based rank).

    Chunks are identified by chunk_id (or id); strings are document IDs. Count
    each ID once per input list. Preserve the first representation/metadata and
    break ties by first appearance. Dictionary results gain rrf_score; inputs
    and their original retrieval scores are never modified.
    """
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 0:
        raise ValueError("k must be a finite, non-negative number")
    scores: dict[str, float] = {}
    documents: dict[str, str | dict[str, Any]] = {}
    for ranked in ranked_lists:
        seen = set()
        for rank, document in enumerate(ranked, start=1):
            if isinstance(document, str):
                document_id = document
            elif isinstance(document, dict):
                document_id = document.get("chunk_id") or document.get("id")
            else:
                raise ValueError("Documents must be string IDs or dictionaries")
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("Each document must have a non-empty string ID")
            if document_id in seen:
                continue
            seen.add(document_id)
            documents.setdefault(document_id, document)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (rank + k)
    ordered_ids = sorted(scores, key=lambda document_id: -scores[document_id])
    return [
        {**documents[document_id], "rrf_score": scores[document_id]}
        if isinstance(documents[document_id], dict) else documents[document_id]
        for document_id in ordered_ids
    ]
