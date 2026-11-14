"""Synchronous PDF extraction and page-aware token/section chunking.

Call process_pdf through a thread pool from async routes/background pipelines.
"""

from bisect import bisect_left, bisect_right
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import fitz
import logfire
import tiktoken

from core.config import settings
from core.limits import CHUNK_MAX_TEXT_JSON_BYTES, EMBEDDING_MAX_INPUT_TOKENS
from utils.text_utils import clean_text, count_tokens, extract_section_headers


def extract_text_from_pdf(file_path: str | Path) -> dict[str, Any]:
    """Extract raw text with 1-based PDF page numbers, preserving line breaks.

    Scanned/image-only documents require OCR, which this processor does not do.
    PyMuPDF exceptions for corrupt or missing documents propagate to the caller.
    """
    with logfire.span("pdf.extract") as span:
        with fitz.open(file_path) as document:
            if not document.is_pdf:
                raise ValueError("The document is not a PDF")
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            pages = [
                {"page_num": number, "text": page.get_text()}
                for number, page in enumerate(document, start=1)
            ]
        if not any(clean_text(page["text"]) for page in pages):
            raise ValueError("PDF has no extractable text; scanned PDFs require OCR")
        span.set_attribute("page_count", len(pages))
        return {
            "full_text": "\n\n".join(page["text"] for page in pages),
            "pages": pages,
            "page_count": len(pages),
        }


def _make_chunk(text: str, page_nums: list[int], index: int) -> dict[str, Any]:
    return {
        "text": text,
        "page_num": page_nums[0],
        "page_nums": list(page_nums),
        "chunk_index": index,
        "chunk_id": str(uuid4()),
    }


def chunk_text_fixed(
    pages: list[dict[str, Any]], chunk_size: int, overlap: int
) -> list[dict[str, Any]]:
    """Split each page into token windows; never merge different PDF pages.

    Unicode-safe token boundaries can slightly reduce window size or overlap.
    Raises ValueError if the budget cannot fit even one complete character.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    encoding = tiktoken.encoding_for_model(settings.OPENAI_EMBEDDING_MODEL)
    chunks = []
    for page in pages:
        text = clean_text(page["text"])
        if not text:
            continue
        tokens = encoding.encode(text, disallowed_special=())
        # A token may end inside a UTF-8 character. Only split at boundaries
        # shared by tokens and Unicode characters to avoid corrupting text.
        raw = text.encode("utf-8")
        offset = 0
        boundaries = [0]
        for index, token in enumerate(tokens, start=1):
            offset += len(encoding.decode_single_token_bytes(token))
            if offset == len(raw) or raw[offset] & 0xC0 != 0x80:
                boundaries.append(index)
        start = 0
        while start < len(tokens):
            end_index = bisect_right(boundaries, start + chunk_size) - 1
            end = boundaries[end_index]
            while end > start:
                chunk_text = encoding.decode(tokens[start:end], errors="strict")
                if len(encoding.encode(chunk_text, disallowed_special=())) <= chunk_size:
                    break
                end_index -= 1
                end = boundaries[end_index]
            if end <= start:
                raise ValueError("chunk_size is too small for a complete Unicode character")
            if chunk_text.strip():
                chunks.append(_make_chunk(chunk_text, [page["page_num"]], len(chunks)))
            if end == len(tokens):
                break
            start = boundaries[bisect_left(boundaries, max(start + 1, end - overlap))]
    return chunks


def _text_fits_metadata(text: str) -> bool:
    return len(json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")) <= CHUNK_MAX_TEXT_JSON_BYTES


def _chunk_budget() -> tuple[int, int]:
    if settings.CHUNK_SIZE <= 0 or settings.CHUNK_OVERLAP < 0 or settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        raise ValueError("Require CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
    size = min(settings.CHUNK_SIZE, EMBEDDING_MAX_INPUT_TOKENS)
    return size, min(settings.CHUNK_OVERLAP, size - 1)


def _bounded_fixed_chunks(pages: list[dict[str, Any]], size: int, overlap: int) -> list[dict[str, Any]]:
    """Reduce the token window if text is unusually large after JSON encoding."""
    while True:
        chunks = chunk_text_fixed(pages, size, min(overlap, size - 1))
        if all(_text_fits_metadata(chunk["text"]) for chunk in chunks):
            return chunks
        if size <= 1:
            raise ValueError("Chunk text cannot fit the metadata byte budget")
        size = max(1, size // 2)


def chunk_text_semantic(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep short sections whole; split long ones into bounded page windows.

    Both heading-based and fallback chunks respect the configured token budget,
    the embedding input ceiling and the metadata byte budget. No text is
    truncated. Split sections carry the actual page of each individual window.
    """
    size, overlap = _chunk_budget()
    chunks = []
    section_pages: list[dict[str, Any]] = []
    found_header = False

    def flush() -> None:
        text = clean_text("\n".join(page["text"] for page in section_pages))
        if text:
            if count_tokens(text, settings.OPENAI_EMBEDDING_MODEL) <= size and _text_fits_metadata(text):
                chunks.append(_make_chunk(text, [page["page_num"] for page in section_pages], len(chunks)))
            else:
                for chunk in _bounded_fixed_chunks(section_pages, size, overlap):
                    chunk["chunk_index"] = len(chunks)
                    chunks.append(chunk)
        section_pages.clear()

    for page in pages:
        for line in page["text"].splitlines():
            if not clean_text(line):
                continue
            if extract_section_headers(line):
                found_header = True
                flush()
            if section_pages and section_pages[-1]["page_num"] == page["page_num"]:
                section_pages[-1]["text"] += "\n" + line
            else:
                section_pages.append({"page_num": page["page_num"], "text": line})
    if not found_header:
        return _bounded_fixed_chunks(pages, size, overlap)
    flush()
    return chunks


def process_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract a PDF and return its semantic (or fallback fixed) chunks."""
    with logfire.span("pdf.process"):
        return process_pdf_document(file_path)["chunks"]


def process_pdf_document(file_path: str | Path) -> dict[str, Any]:
    """Return extracted text, pages, page count and chunks from one PDF read."""
    with logfire.span("pdf.process_document") as span:
        document = extract_text_from_pdf(file_path)
        document["chunks"] = chunk_text_semantic(document["pages"])
        span.set_attribute("page_count", document["page_count"])
        span.set_attribute("chunk_count", len(document["chunks"]))
        return document
