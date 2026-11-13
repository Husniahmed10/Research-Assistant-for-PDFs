"""Run from backend: python -m utils.test_health_errors (cloud probes mocked)."""

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes import health
from core.config import settings
from core.exceptions import ErrorHandlingRoute, internal_error_handler
from db import crud
from services import portkey_client
from utils.test_papers import PaperAPIFixture, main


class HealthTests(PaperAPIFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.index.describe_index_stats = Mock(return_value={"total_vector_count": 0})
        self.gateway = patch.object(portkey_client, "chat_completion", AsyncMock(return_value="OK"))
        self.gateway_mock = self.gateway.start()
        self.addCleanup(self.gateway.stop)

    async def test_healthy_checks_actual_sql_and_both_remote_probes(self):
        response = await self.client.get("/api/v1/health/")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual({key: data[key] for key in ("status", "db", "pinecone", "llm")},
                         {"status": "healthy", "db": "ok", "pinecone": "ok", "llm": "ok"})
        self.assertIsNotNone(datetime.fromisoformat(data["timestamp"]).tzinfo)
        self.index.describe_index_stats.assert_called_once()
        self.gateway_mock.assert_awaited_once()
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual((await self.client.get("/api/v1/health")).status_code, 200)

    async def test_independent_failure_results_are_sanitized(self):
        cases = [
            ("db", patch.object(AsyncSession, "execute", AsyncMock(side_effect=RuntimeError("private DB")))),
            ("pinecone", patch.object(self.index, "describe_index_stats", side_effect=RuntimeError("private API key"))),
            ("llm", patch.object(portkey_client, "chat_completion", AsyncMock(side_effect=RuntimeError("private provider")))),
        ]
        for failed, replacement in cases:
            with self.subTest(failed=failed), replacement:
                response = await self.client.get("/api/v1/health/")
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json()["status"], "degraded")
            self.assertEqual(response.json()[failed], "error")
            self.assertNotIn("private", response.text)
            for dependency in {"db", "pinecone", "llm"} - {failed}:
                self.assertEqual(response.json()[dependency], "ok")

    async def test_missing_index_and_llm_timeout_are_degraded(self):
        with patch.object(main.app.state, "pinecone_index", None):
            response = await self.client.get("/api/v1/health/")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["pinecone"], "error")
        cancelled = asyncio.Event()
        async def slow(*args):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()
        with patch.object(health, "_CHECK_TIMEOUT_SECONDS", 0.05), \
                patch.object(health, "check_db", AsyncMock(return_value="ok")), \
                patch.object(health, "check_pinecone", AsyncMock(return_value="ok")), \
                patch.object(portkey_client, "chat_completion", AsyncMock(side_effect=slow)):
            response = await asyncio.wait_for(self.client.get("/api/v1/health/"), 2)
        self.assertEqual(response.json()["llm"], "error")
        self.assertEqual(response.status_code, 503)
        self.assertTrue(cancelled.is_set())

    async def test_checks_are_concurrent_and_liveness_does_not_call_gateway(self):
        self.assertEqual((await self.client.get("/")).status_code, 200)
        self.gateway_mock.assert_not_awaited()
        arrived = set()
        all_started = asyncio.Event()
        def checker(name):
            async def check(*args):
                arrived.add(name)
                if len(arrived) == 3:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), 2)
                return "ok"
            return check
        with patch.object(health, "check_db", side_effect=checker("db")), \
                patch.object(health, "check_pinecone", side_effect=checker("pinecone")), \
                patch.object(health, "check_llm", side_effect=checker("llm")):
            response = await self.client.get("/api/v1/health/")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(arrived), 3)


class ErrorHandlingTests(PaperAPIFixture):
    async def test_unexpected_route_errors_are_500_logged_and_keep_cors(self):
        with patch.object(crud, "get_all_papers", side_effect=RuntimeError("private API key")), \
                patch("core.exceptions.logfire.error") as logged:
            response = await self.client.get("/api/v1/papers/", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error"})
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")
        logged.assert_called_once()
        self.assertEqual(logged.call_args.kwargs["error_type"], "RuntimeError")
        self.assertNotIn("private API", str(logged.call_args))
        # The same boundary covers SSE preflight and evaluation handlers.
        with patch.object(crud, "get_session", side_effect=RuntimeError("private DB")):
            response = await self.client.get(f"/api/v1/chat/sessions/{uuid4()}/stream", params={"query": "Q"})
        self.assertEqual(response.status_code, 500)
        with patch.object(crud, "get_paper", side_effect=RuntimeError("private DB")):
            response = await self.client.get(f"/api/v1/evaluation/papers/{uuid4()}/report")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private", response.text)

    async def test_validation_bad_upload_not_found_and_route_coverage(self):
        response = await self.client.post("/api/v1/papers/upload", files={"file": ("bad.pdf", b"invalid", "application/pdf")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual((await self.client.get(f"/api/v1/papers/{uuid4()}/status")).status_code, 404)
        self.assertEqual((await self.client.get("/unknown-route")).status_code, 404)
        response = await self.client.get("/api/v1/papers/not-a-uuid/status")
        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.json()["detail"])
        self.assertNotIn("input", response.json()["detail"][0])
        for route in main.app.routes:
            if getattr(route, "path", "").startswith("/api/v1/"):
                self.assertIsInstance(route, ErrorHandlingRoute)

    async def test_global_fallback_handler_also_protects_non_api_routes(self):
        self.assertIs(main.app.exception_handlers[Exception], internal_error_handler)
        app = FastAPI(exception_handlers={Exception: main.app.exception_handlers[Exception]})
        async def crash(request):
            raise RuntimeError("private failure")
        app.add_route("/crash", crash)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                     base_url="http://test") as client:
            response = await client.get("/crash")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error"})


class StartupOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_db_index_bm25_uploads_order_and_no_duplicate_routers(self):
        calls = []
        index = object()
        dispose = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            uploads = Path(directory) / "uploads"
            async def init_db():
                self.assertFalse(uploads.exists())
                calls.append("db")
            def init_index():
                self.assertFalse(uploads.exists())
                calls.append("pinecone")
                return index
            def load_bm25():
                self.assertFalse(uploads.exists())
                self.assertIs(main.app.state.pinecone_index, index)
                calls.append("bm25")
            route_count = len(main.app.routes)
            with patch.object(settings, "UPLOAD_DIR", str(uploads)), \
                    patch.object(main, "init_db", side_effect=init_db), \
                    patch.object(main, "init_pinecone_index", side_effect=init_index), \
                    patch.object(main, "load_bm25_encoders", side_effect=load_bm25), \
                    patch.object(main, "engine", SimpleNamespace(dispose=dispose)):
                async with main.app.router.lifespan_context(main.app):
                    self.assertTrue(uploads.is_dir())
                    self.assertEqual(calls, ["db", "pinecone", "bm25"])
                    self.assertEqual(len(main.app.routes), route_count)
            self.assertIsNone(main.app.state.pinecone_index)
            dispose.assert_awaited_once()

    async def test_bm25_failure_stops_startup_and_disposes_engine(self):
        dispose = AsyncMock()
        with patch.object(main, "init_db", AsyncMock()), \
                patch.object(main, "init_pinecone_index", return_value=object()), \
                patch.object(main, "load_bm25_encoders", side_effect=ValueError("Invalid BM25 parameters")), \
                patch.object(main, "engine", SimpleNamespace(dispose=dispose)):
            with self.assertRaisesRegex(ValueError, "Invalid BM25"):
                async with main.app.router.lifespan_context(main.app):
                    self.fail("Must not start with invalid saved encoder parameters")
        self.assertIsNone(main.app.state.pinecone_index)
        dispose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
