"""Run from backend: python -m utils.test_processing.

Uses temporary PDFs and a mocked HTTP transport; no paid API requests are made.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import fitz
import httpx
import tiktoken
from openai import AsyncOpenAI, RateLimitError

from core.config import settings
from services import embedder
from services.pdf_processor import (
    chunk_text_fixed, chunk_text_semantic, extract_text_from_pdf, process_pdf,
)
from utils.text_utils import clean_text, count_tokens


class PDFTests(unittest.TestCase):
    def test_real_pdf_sections_and_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            with fitz.open() as document:
                document.new_page().insert_text(
                    (72, 72), "Paper title\nAbstract\nSummary.\n1. Introduction\nStart of introduction."
                )
                document.new_page().insert_text(
                    (72, 72), "Continuation of introduction.\n2. Results\nAccuracy is 95 percent."
                )
                document.new_page()
                document.save(path)
            extracted = extract_text_from_pdf(path)
            self.assertEqual(extracted["page_count"], 3)
            self.assertEqual([p["page_num"] for p in extracted["pages"]], [1, 2, 3])
            self.assertIn("Accuracy is 95 percent", extracted["full_text"])
            chunks = process_pdf(path)
            self.assertEqual(len(chunks), 4)
            self.assertEqual(chunks[0]["text"], "Paper title")
            self.assertEqual(chunks[2]["page_nums"], [1, 2])
            self.assertEqual(chunks[3]["page_num"], 2)
            self.assertEqual([c["chunk_index"] for c in chunks], list(range(4)))
            self.assertEqual(len({UUID(c["chunk_id"]) for c in chunks}), 4)

    def test_fixed_chunks_overlap_and_fallback(self):
        text = " ".join(f"word{i}" for i in range(80))
        pages = [{"page_num": 2, "text": text}, {"page_num": 3, "text": "  "}]
        encoding = tiktoken.encoding_for_model(settings.OPENAI_EMBEDDING_MODEL)
        chunks = chunk_text_fixed(pages, 30, 5)
        tokens = encoding.encode(clean_text(text))
        expected = []
        for start in range(0, len(tokens), 25):
            expected.append(encoding.decode(tokens[start:start + 30]))
            if start + 30 >= len(tokens):
                break
        self.assertEqual([c["text"] for c in chunks], expected)
        self.assertTrue(all(c["page_num"] == 2 for c in chunks))
        with patch.object(settings, "CHUNK_SIZE", 30), patch.object(settings, "CHUNK_OVERLAP", 5):
            self.assertEqual([c["text"] for c in chunk_text_semantic(pages)], expected)
        for size, overlap in ((0, 0), (10, -1), (10, 10)):
            with self.assertRaises(ValueError):
                chunk_text_fixed(pages, size, overlap)

    def test_unicode_no_loss_and_empty_pdf(self):
        text = "研究の結果 🌍 café සිංහල " * 15
        chunks = chunk_text_fixed([{"page_num": 1, "text": text}], 16, 0)
        self.assertEqual("".join(c["text"] for c in chunks), clean_text(text))
        self.assertTrue(all(count_tokens(c["text"], settings.OPENAI_EMBEDDING_MODEL) <= 16 for c in chunks))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.pdf"
            with fitz.open() as document:
                document.new_page()
                document.save(path)
            with self.assertRaisesRegex(ValueError, "no extractable text"):
                extract_text_from_pdf(path)
            encrypted = Path(directory) / "encrypted.pdf"
            with fitz.open() as document:
                document.new_page().insert_text((72, 72), "Secret")
                document.save(encrypted, encryption=fitz.PDF_ENCRYPT_AES_256,
                              owner_pw="owner", user_pw="reader")
            with self.assertRaisesRegex(ValueError, "Password-protected"):
                extract_text_from_pdf(encrypted)
        self.assertEqual(chunk_text_semantic([]), [])


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.mode = "normal"

        def respond(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            if self.mode == "rate_limit":
                return httpx.Response(429, json={"error": {"message": "Test rate limit", "type": "rate_limit_error"}})
            data = [
                {"object": "embedding", "index": i, "embedding": [float(text), 0.5]}
                for i, text in enumerate(payload["input"])
            ]
            if self.mode == "missing":
                data = data[:-1]
            return httpx.Response(200, json={
                "object": "list", "model": settings.OPENAI_EMBEDDING_MODEL,
                "data": list(reversed(data)), "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })

        self.client = AsyncOpenAI(
            api_key="test-key", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        self.client_patch = patch.object(embedder, "client", self.client)
        self.key_patch = patch.object(settings, "OPENAI_API_KEY", "test-key")
        self.client_patch.start()
        self.key_patch.start()

    async def asyncTearDown(self):
        self.client_patch.stop()
        self.key_patch.stop()
        await self.client.close()

    async def test_batch_boundaries_order_and_chunks(self):
        vectors = await embedder.embed_batch([str(i) for i in range(205)])
        self.assertEqual([len(r["input"]) for r in self.requests], [100, 100, 5])
        self.assertEqual([v[0] for v in vectors], list(range(205)))
        self.assertEqual(await embedder.embed_text("42"), [42.0, 0.5])
        self.assertEqual(self.requests[-1]["model"], settings.OPENAI_EMBEDDING_MODEL)
        self.assertEqual(self.requests[-1]["dimensions"], settings.PINECONE_DIMENSION)
        chunks = [{"text": "1", "page_num": 2, "chunk_id": "abc"}]
        self.assertIs(await embedder.embed_chunks(chunks), chunks)
        self.assertEqual(chunks[0]["embedding"], [1.0, 0.5])
        self.assertEqual(chunks[0]["chunk_id"], "abc")

    async def test_empty_invalid_and_request_token_budget(self):
        self.assertEqual(await embedder.embed_batch([]), [])
        self.assertEqual(await embedder.embed_chunks([]), [])
        for texts in (["1", " "], [None]):
            with self.assertRaises(ValueError):
                await embedder.embed_batch(texts)
        with patch.object(embedder, "count_tokens", return_value=8193):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                await embedder.embed_text("1")
        self.assertEqual(self.requests, [])
        with patch.object(embedder, "count_tokens", return_value=8192):
            await embedder.embed_batch([str(i) for i in range(40)])
        self.assertEqual([len(r["input"]) for r in self.requests], [36, 4])

    async def test_failed_responses_do_not_mutate_chunks(self):
        self.mode = "missing"
        with self.assertRaises(RuntimeError):
            await embedder.embed_batch(["1", "2"])
        self.mode = "rate_limit"
        chunks = [{"text": "1"}]
        with self.assertRaises(RateLimitError):
            await embedder.embed_chunks(chunks)
        self.assertNotIn("embedding", chunks[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
