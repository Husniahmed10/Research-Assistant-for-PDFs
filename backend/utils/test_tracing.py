"""Run from backend: python -m utils.test_tracing (local in-memory export only)."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from weakref import WeakSet

import fitz
import httpx
import logfire
from fastapi import FastAPI
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from core.config import settings
from tracing import langsmith_setup, logfire_setup


class SetupTests(unittest.TestCase):
    def test_langsmith_enabled_disabled_and_missing_key(self):
        for switch, key, expected in ((True, "test-only", "true"), (False, "test-only", "false"),
                                      (True, "", "false")):
            with self.subTest(switch=switch, has_key=bool(key)), patch.dict(os.environ, {}, clear=True), \
                    patch.object(settings, "LANGCHAIN_TRACING_V2", switch), \
                    patch.object(settings, "LANGCHAIN_API_KEY", key):
                langsmith_setup.setup_langsmith()
                self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], expected)
                self.assertEqual(os.environ["LANGCHAIN_API_KEY"], key)
                self.assertEqual(os.environ["LANGCHAIN_PROJECT"], settings.LANGCHAIN_PROJECT)
                self.assertEqual(os.environ["LANGCHAIN_ENDPOINT"], settings.LANGCHAIN_ENDPOINT)

    def test_logfire_idempotence_and_no_token_disables_export(self):
        app = FastAPI()
        with patch.object(settings, "LOGFIRE_TOKEN", ""), \
                patch.object(logfire_setup, "_configured", False), \
                patch.object(logfire_setup, "_httpx_instrumented", False), \
                patch.object(logfire_setup, "_engines", WeakSet()), \
                patch.object(logfire, "configure") as configure, \
                patch.object(logfire, "instrument_fastapi") as fastapi, \
                patch.object(logfire, "instrument_sqlalchemy") as sqlalchemy, \
                patch.object(logfire, "instrument_httpx") as http:
            logfire_setup.setup_logfire(app)
            logfire_setup.setup_logfire(app)
            logfire_setup.setup_logfire(FastAPI())
        configure.assert_called_once()
        self.assertFalse(configure.call_args.kwargs["send_to_logfire"])
        sqlalchemy.assert_called_once_with(engine=logfire_setup.engine.sync_engine)
        http.assert_called_once()
        self.assertEqual(fastapi.call_count, 2)


class LocalInstrumentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_sql_and_service_spans(self):
        from agents.nodes.synthesizer import synthesizer_node
        from services import embedder, pdf_processor, pinecone_client, portkey_client

        exporter = InMemorySpanExporter()
        logfire.configure(send_to_logfire=False, console=False,
                          additional_span_processors=[SimpleSpanProcessor(exporter)])
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        app = FastAPI()

        @app.get("/health")
        async def health():
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return {"status": "ok"}

        try:
            with patch.object(logfire_setup, "engine", engine), \
                    patch.object(logfire_setup, "_configured", True), \
                    patch.object(logfire_setup, "_httpx_instrumented", False), \
                    patch.object(logfire_setup, "_engines", WeakSet()):
                logfire_setup.setup_logfire(app)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                self.assertEqual((await client.get("/health")).json(), {"status": "ok"})

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sample.pdf"
                with fitz.open() as document:
                    document.new_page().insert_text((72, 72), "Abstract\nA study.\nConclusion\nIt works.")
                    document.save(path)
                self.assertTrue(pdf_processor.process_pdf(path))

            response = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
                                       usage=SimpleNamespace(total_tokens=4))
            with patch.object(settings, "OPENAI_API_KEY", "test-only"), \
                    patch.object(embedder.client.embeddings, "create", AsyncMock(return_value=response)):
                self.assertEqual(await embedder.embed_text("Test text"), [0.1, 0.2])

            index = SimpleNamespace(query=lambda **kwargs: {"matches": [{"id": "chunk-a", "score": 0.9,
                "metadata": {"paper_id": "paper-a", "page_num": 1, "text": "It works."}}]})
            pinecone_client.dense_search(index, [0.1] * settings.PINECONE_DIMENSION, "paper-a", 5)
            with patch.object(portkey_client, "chat_completion_with_metadata", AsyncMock(return_value={
                "answer": "It works [1].", "model_used": "groq-test-model",
            })):
                await synthesizer_node({"query": "What is the conclusion?", "steps": [],
                    "reranked_chunks": [{"text": "It works.", "page_num": 1, "chunk_id": "chunk-a"}]})
            spans = exporter.get_finished_spans()
            names = {span.name for span in spans}
            self.assertTrue(any("GET /health" in name for name in names), names)
            self.assertTrue(any("SELECT" in name for name in names), names)
            for name in ("pdf.extract", "pdf.process", "pdf.process_document", "embedding.batch",
                         "pinecone.search", "agent.synthesize"):
                self.assertIn(name, names)
            attrs = {span.name: span.attributes for span in spans}
            self.assertEqual(attrs["embedding.batch"]["total_tokens"], 4)
            self.assertEqual(attrs["pinecone.search"]["result_count"], 1)
            self.assertGreaterEqual(attrs["pinecone.search"]["latency_ms"], 0)
            self.assertEqual(attrs["agent.synthesize"]["model_used"], "groq-test-model")
            self.assertGreaterEqual(attrs["agent.synthesize"]["latency_ms"], 0)
        finally:
            await engine.dispose()
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            SQLAlchemyInstrumentor().uninstrument()
            HTTPXClientInstrumentor().uninstrument()
            FastAPIInstrumentor.uninstrument_app(app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
