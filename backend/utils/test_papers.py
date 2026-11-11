"""Run from backend: python -m utils.test_papers.

API integration tests with temporary PDFs/SQLite and mocked summary generation.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import fitz
import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import main
from api.routes import papers
from core.config import settings
from core.dependencies import get_db, get_pinecone_index
from db import crud, database
from services.summarizer import SUMMARY_KEYS
from utils.test_pinecone import FakeIndex
from starlette.requests import Request


class PaperAPIFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.uploads = Path(self.directory.name) / "uploads"
        db_path = (Path(self.directory.name) / "test.sqlite").as_posix()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.summary = {key: f"Summary for {key}" for key in SUMMARY_KEYS}
        self.summary_mock = AsyncMock(return_value=self.summary)
        self.index = FakeIndex()
        self.index.upsert = Mock(wraps=self.index.upsert)
        self.index.delete = Mock(wraps=self.index.delete)

        async def fake_embeddings(chunks):
            return [{**chunk, "embedding": [0.1] * settings.PINECONE_DIMENSION} for chunk in chunks]

        self.embedding_mock = AsyncMock(side_effect=fake_embeddings)
        self.patches = [
            patch.object(settings, "UPLOAD_DIR", str(self.uploads)),
            patch.object(database, "engine", self.engine),
            patch.object(main, "engine", self.engine),
            patch.object(papers, "AsyncSessionLocal", self.sessions),
            patch.object(papers, "generate_summary", self.summary_mock),
            patch.object(papers, "embed_chunks", self.embedding_mock),
            patch.object(main, "init_pinecone_index", return_value=self.index),
        ]
        for item in self.patches:
            item.start()

        async def test_db():
            async with self.sessions() as db:
                yield db

        main.app.dependency_overrides[get_db] = test_db
        self.lifespan = main.app.router.lifespan_context(main.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False),
            base_url="http://test",
        )
        with fitz.open() as document:
            document.new_page().insert_text((72, 72), "Abstract\nA sample paper.\nConclusion\nIt works.")
            self.pdf = document.tobytes()

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        main.app.dependency_overrides.pop(get_db, None)
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    async def upload(self, data=None, mime="application/pdf"):
        return await self.client.post(
            "/api/v1/papers/upload",
            files={"file": ("example.pdf", self.pdf if data is None else data, mime)},
        )


class PaperAPITests(PaperAPIFixture):
    async def test_upload_summary_list_and_delete(self):
        response = await self.upload()
        self.assertEqual(response.status_code, 202, response.text)
        result = response.json()
        self.assertEqual(result["status"], "processing")
        paper_id = result["paper_id"]
        prefix = f"/api/v1/papers/{paper_id}"
        self.assertEqual((await self.client.get(f"{prefix}/status")).json(), {
            "paper_id": paper_id, "status": "ready", "page_count": 1,
        })
        self.assertEqual((await self.client.get(f"{prefix}/summary")).json(), self.summary)
        self.summary_mock.assert_awaited_once()
        self.assertIn("It works.", self.summary_mock.call_args.args[0])
        self.embedding_mock.assert_awaited_once()
        self.assertTrue(self.index.records)
        self.assertTrue(all(row["metadata"]["paper_id"] == paper_id for row in self.index.records.values()))
        self.assertTrue(all(len(row["values"]) == settings.PINECONE_DIMENSION for row in self.index.records.values()))
        listing = (await self.client.get("/api/v1/papers/")).json()
        self.assertEqual([row["id"] for row in listing], [paper_id])
        self.assertNotIn("file_path", listing[0])
        async with self.sessions() as db:
            paper = await crud.get_paper(db, paper_id)
            path = Path(paper.file_path)
            self.assertEqual(path.name, f"{paper_id}_example.pdf")
            self.assertIn("It works.", paper.full_text)
            self.assertEqual(json.loads(paper.summary_json), self.summary)
            # Also verify deleting through the API cascades related records.
            chat = await crud.create_session(db, paper_id, "Test")
            await crud.add_message(db, chat.id, "user", "Question")
            chat_id = chat.id
            await db.commit()
        self.assertEqual((await self.client.delete(prefix)).json(), {"message": "deleted"})
        self.assertFalse(path.exists())
        self.assertEqual(self.index.records, {})
        self.index.delete.assert_called_once_with(filter={"paper_id": paper_id})
        self.assertEqual((await self.client.get(f"{prefix}/status")).status_code, 404)
        self.assertEqual((await self.client.get("/api/v1/papers/")).json(), [])
        async with self.sessions() as db:
            self.assertEqual(await crud.get_messages_for_session(db, chat_id), [])

    async def test_invalid_uploads_have_no_side_effects(self):
        for data, mime in ((b"not a pdf", "application/pdf"), (self.pdf, "text/plain")):
            self.assertEqual((await self.upload(data, mime)).status_code, 400)
        with patch.object(settings, "MAX_FILE_SIZE_MB", 0):
            self.assertEqual((await self.upload()).status_code, 400)
        self.assertEqual((await self.client.get("/api/v1/papers/")).json(), [])
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.summary_mock.assert_not_awaited()
        self.embedding_mock.assert_not_awaited()
        self.index.upsert.assert_not_called()

    async def test_background_failures_are_persisted(self):
        self.summary_mock.side_effect = RuntimeError("private provider error")
        with self.assertLogs(papers.logger, level="ERROR"):
            response = await self.upload()
        self.assertEqual(response.status_code, 202)
        paper_id = response.json()["paper_id"]
        prefix = f"/api/v1/papers/{paper_id}"
        self.assertEqual((await self.client.get(f"{prefix}/status")).json()["status"], "failed")
        response = await self.client.get(f"{prefix}/summary")
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("private provider error", response.text)
        self.assertEqual((await self.client.delete(prefix)).status_code, 200)

    async def test_corrupt_pdf_marks_failed(self):
        with self.assertLogs(papers.logger, level="ERROR"):
            response = await self.upload(b"%PDF-broken document")
        self.assertEqual(response.status_code, 202)
        paper_id = response.json()["paper_id"]
        status = await self.client.get(f"/api/v1/papers/{paper_id}/status")
        self.assertEqual(status.json()["status"], "failed")
        self.summary_mock.assert_not_awaited()

    async def test_processing_conflicts_and_missing_papers(self):
        path = self.uploads / "processing.pdf"
        path.write_bytes(self.pdf)
        async with self.sessions() as db:
            paper = await crud.create_paper(db, "processing.pdf", str(path), 1, status="processing")
            paper_id = paper.id
            await db.commit()
        prefix = f"/api/v1/papers/{paper_id}"
        self.assertEqual((await self.client.get(f"{prefix}/summary")).status_code, 409)
        self.assertEqual((await self.client.delete(prefix)).status_code, 409)
        self.assertTrue(path.exists())
        missing = f"/api/v1/papers/{uuid4()}"
        for suffix in ("/status", "/summary"):
            self.assertEqual((await self.client.get(missing + suffix)).status_code, 404)
        self.assertEqual((await self.client.delete(missing)).status_code, 404)

    async def test_database_failure_cleans_saved_file(self):
        with patch.object(papers.crud, "create_paper", AsyncMock(side_effect=RuntimeError("DB unavailable"))):
            response = await self.upload()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.assertEqual((await self.client.get("/api/v1/papers/")).json(), [])
        self.summary_mock.assert_not_awaited()

    async def test_delete_failure_preserves_record_for_retry(self):
        response = await self.upload()
        paper_id = response.json()["paper_id"]
        prefix = f"/api/v1/papers/{paper_id}"
        with patch.object(papers, "delete_upload_file", side_effect=PermissionError("locked")):
            self.assertEqual((await self.client.delete(prefix)).status_code, 500)
        self.assertEqual((await self.client.get(f"{prefix}/status")).status_code, 200)
        self.assertEqual((await self.client.delete(prefix)).status_code, 200)

    async def test_dependency_returns_startup_index_or_503(self):
        request = Request({"type": "http", "app": main.app})
        self.assertIs(get_pinecone_index(request), self.index)
        with patch.object(main.app.state, "pinecone_index", None):
            self.assertEqual((await self.upload()).status_code, 503)
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.summary_mock.assert_not_awaited()

    async def test_embedding_failure_never_marks_ready(self):
        self.embedding_mock.side_effect = RuntimeError("Embedding unavailable")
        with self.assertLogs(papers.logger, level="ERROR"):
            response = await self.upload()
        paper_id = response.json()["paper_id"]
        self.assertEqual((await self.client.get(f"/api/v1/papers/{paper_id}/status")).json()["status"], "failed")
        self.index.upsert.assert_not_called()

    async def test_partial_index_failure_cleans_vectors(self):
        def fail_after_partial_write(vectors):
            FakeIndex.upsert(self.index, vectors[:1])
            raise RuntimeError("Index unavailable")

        self.index.upsert.side_effect = fail_after_partial_write
        with self.assertLogs(papers.logger, level="ERROR"):
            response = await self.upload()
        paper_id = response.json()["paper_id"]
        prefix = f"/api/v1/papers/{paper_id}"
        self.assertEqual((await self.client.get(f"{prefix}/status")).json()["status"], "failed")
        self.assertEqual((await self.client.get(f"{prefix}/summary")).status_code, 409)
        self.index.delete.assert_called_once_with(filter={"paper_id": paper_id})
        self.assertEqual(self.index.records, {})

    async def test_vector_delete_failure_preserves_file_and_record(self):
        response = await self.upload()
        paper_id = response.json()["paper_id"]
        prefix = f"/api/v1/papers/{paper_id}"
        async with self.sessions() as db:
            path = Path((await crud.get_paper(db, paper_id)).file_path)
        self.index.delete.side_effect = RuntimeError("Index unavailable")
        self.assertEqual((await self.client.delete(prefix)).status_code, 503)
        self.assertTrue(path.exists())
        self.assertEqual((await self.client.get(f"{prefix}/status")).status_code, 200)
        self.index.delete.side_effect = None
        self.assertEqual((await self.client.delete(prefix)).status_code, 200)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
