"""Run from backend: python -m utils.test_summary_app.

Uses mocked model replies, an in-memory database, and temporary upload storage.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

import main
from core.config import settings
from db import database
from services import summarizer
from utils.text_utils import count_tokens


def sample_summary():
    return {key: f"Text for {key}" for key in summarizer.SUMMARY_KEYS}


def reply(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


class ParserTests(unittest.TestCase):
    def test_json_and_regex_fallbacks(self):
        summary = sample_summary()
        raw = json.dumps(summary)
        for response in (raw, f"```json\n{raw}\n```", f"Here is the summary:\n{raw}"):
            self.assertEqual(summarizer.parse_summary_response(response), summary)
        headings = "\n\n".join(
            f"## {key.replace('_', ' ').title()}\n{value}"
            for key, value in summary.items()
        )
        self.assertEqual(summarizer.parse_summary_response(headings), summary)
        bold = "\n".join(f"**{key.replace('_', ' ').title()}:** {value}" for key, value in summary.items())
        self.assertEqual(summarizer.parse_summary_response(bold), summary)

    def test_invalid_summaries_are_rejected(self):
        for response in ("", "not a summary", "{}", "[]", '{"abstract": "unfinished',
                         json.dumps({**sample_summary(), "abstract": []}),
                         json.dumps({**sample_summary(), "abstract": " "})):
            with self.subTest(response=response), self.assertRaises(ValueError):
                summarizer.parse_summary_response(response)


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_with_direct_client(self):
        create = AsyncMock(return_value=reply(json.dumps(sample_summary())))
        with patch.object(settings, "OPENAI_API_KEY", "test-key"), \
                patch.object(summarizer.client.chat.completions, "create", create):
            self.assertEqual(await summarizer.generate_summary("Paper content."), sample_summary())
        payload = create.call_args.kwargs
        self.assertEqual(payload["model"], settings.OPENAI_MODEL)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("Paper content.", payload["messages"][1]["content"])

    async def test_input_is_truncated(self):
        create = AsyncMock(return_value=reply(json.dumps(sample_summary())))
        with patch.object(settings, "OPENAI_API_KEY", "test-key"), \
                patch.object(summarizer.client.chat.completions, "create", create), \
                self.assertLogs(summarizer.logger, level="WARNING"):
            await summarizer.generate_summary(" research" * 13_000)
        prompt = create.call_args.kwargs["messages"][1]["content"]
        text = prompt.removeprefix("Paper text: ").split("\n\nReturn a JSON", 1)[0]
        self.assertLessEqual(count_tokens(text, settings.OPENAI_MODEL), 12_000)

    async def test_incomplete_refused_and_empty_replies(self):
        create = AsyncMock()
        with patch.object(settings, "OPENAI_API_KEY", "test-key"), \
                patch.object(summarizer.client.chat.completions, "create", create):
            for response in (reply("{}", "length"), reply(None, refusal="Declined"),
                             reply(""), SimpleNamespace(choices=[])):
                create.return_value = response
                with self.assertRaises(ValueError):
                    await summarizer.generate_summary("Paper content.")
            create.reset_mock()
            with self.assertRaises(ValueError):
                await summarizer.generate_summary(" ")
            create.assert_not_awaited()


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_root_docs_and_cors(self):
        test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")

        async def initialize():
            await database.init_db()
            async with test_engine.connect() as connection:
                tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
                self.assertEqual(set(tables), {"papers", "chat_sessions", "messages"})

        with tempfile.TemporaryDirectory() as directory:
            uploads = Path(directory) / "uploads"
            startup = AsyncMock(side_effect=initialize)
            with patch.object(settings, "UPLOAD_DIR", str(uploads)), \
                    patch.object(database, "engine", test_engine), \
                    patch.object(main, "engine", test_engine), \
                    patch.object(main, "init_db", startup):
                async with main.app.router.lifespan_context(main.app):
                    self.assertTrue(uploads.is_dir())
                    startup.assert_awaited_once()
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
                    ) as client:
                        self.assertEqual((await client.get("/")).json(), {"status": "ok", "app": settings.APP_NAME})
                        self.assertEqual((await client.get("/docs")).status_code, 200)
                        self.assertIn("/", (await client.get("/openapi.json")).json()["paths"])
                        headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
                        response = await client.options("/api/v1/papers/upload", headers=headers)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")
                        headers["Origin"] = "https://unlisted.example"
                        self.assertEqual((await client.options("/", headers=headers)).status_code, 400)
        self.assertEqual(main.papers_router.prefix, "/api/v1/papers")


if __name__ == "__main__":
    unittest.main(verbosity=2)
