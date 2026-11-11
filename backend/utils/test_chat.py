"""Run from backend: python -m utils.test_chat.

Full upload-to-answer API flow with real PDF parsing/SQLite and mocked providers.
"""

import unittest
import re
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from sqlalchemy.exc import SQLAlchemyError

from api.routes import chat
from core.config import settings
from db import crud
from services import simple_rag as rag
from utils.test_papers import PaperAPIFixture


class ChatAPITests(PaperAPIFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        async def answer_from_sources(messages):
            prompt = messages[-1]["content"]
            source = re.search(r"\[(\d+)\] PDF page \d+\nConclusion", prompt)
            label = source.group(1) if source else "1"
            return {"answer": f"It works. [{label}]", "model_used": "groq-test-model"}

        self.chat_mock = AsyncMock(side_effect=answer_from_sources)
        self.chat_patches = [
            patch.object(rag, "embed_text", AsyncMock(return_value=[0.1] * settings.PINECONE_DIMENSION)),
            patch.object(rag.portkey_client, "chat_completion_with_metadata", self.chat_mock),
        ]
        for item in self.chat_patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.chat_patches):
            item.stop()
        await super().asyncTearDown()

    async def ready_session(self):
        upload = await self.upload()
        self.assertEqual(upload.status_code, 202, upload.text)
        paper_id = upload.json()["paper_id"]
        status = await self.client.get(f"/api/v1/papers/{paper_id}/status")
        self.assertEqual(status.json()["status"], "ready")
        session = await self.client.post("/api/v1/chat/sessions", json={"paper_id": paper_id})
        self.assertEqual(session.status_code, 201, session.text)
        return paper_id, session.json()["session_id"]

    async def ask(self, session_id, query="What is the conclusion?"):
        return await self.client.post(f"/api/v1/chat/sessions/{session_id}/message", json={"query": query})

    async def history(self, session_id):
        return await self.client.get(f"/api/v1/chat/sessions/{session_id}/messages")

    async def test_end_to_end_upload_question_and_persisted_history(self):
        paper_id, session_id = await self.ready_session()
        response = await self.ask(session_id)
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertEqual(answer["model_used"], "groq-test-model")
        self.assertIn("It works.", answer["answer"])
        self.assertGreaterEqual(answer["latency_ms"], 0)
        self.assertEqual(len(answer["citations"]), 1)
        self.assertIn(f"[{answer['citations'][0]['id']}]", answer["answer"])
        self.assertEqual(answer["citations"][0]["page_num"], 1)
        self.assertIn("It works.", answer["citations"][0]["text"])
        self.assertEqual((await self.history(session_id)).status_code, 200)
        history = (await self.history(session_id)).json()
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertEqual(history[1]["citations"], answer["citations"])
        self.assertEqual(history[1]["model_used"], "groq-test-model")
        self.assertLess(history[0]["created_at"], history[1]["created_at"])
        self.chat_mock.assert_awaited_once()
        async with self.sessions() as db:
            session = await crud.get_session(db, session_id)
            self.assertEqual(session.paper_id, paper_id)
            self.assertEqual(session.title, "New Chat")

    async def test_only_last_four_messages_of_selected_session(self):
        paper_id, session_id = await self.ready_session()
        other = await self.client.post("/api/v1/chat/sessions", json={"paper_id": paper_id})
        other_id = other.json()["session_id"]
        await self.ask(other_id, "Other session secret")
        answers = []
        for query in ("First question", "Second question", "Third question"):
            response = await self.ask(session_id, query)
            self.assertEqual(response.status_code, 200)
            answers.append(response.json()["answer"])
        with patch.object(chat, "simple_rag", AsyncMock(return_value={
            "answer": "Follow-up", "chunks_used": [], "model_used": "test",
        })) as mocked:
            self.assertEqual((await self.ask(session_id, "Follow up")).status_code, 200)
        self.assertEqual(mocked.call_args.args[1], paper_id)
        history = mocked.call_args.args[2]
        self.assertEqual([message["content"] for message in history], [
            "Second question", answers[1], "Third question", answers[2],
        ])
        self.assertEqual(len((await self.history(session_id)).json()), 8)

    async def test_invalid_and_not_ready_resources(self):
        missing = str(uuid4())
        response = await self.client.post("/api/v1/chat/sessions", json={"paper_id": missing})
        self.assertEqual(response.status_code, 404)
        self.assertEqual((await self.ask(missing)).status_code, 404)
        self.assertEqual((await self.history(missing)).status_code, 404)
        async with self.sessions() as db:
            paper = await crud.create_paper(db, "pending.pdf", "unused", 1, status="processing")
            paper_id = paper.id
            await db.commit()
        self.assertEqual((await self.client.post("/api/v1/chat/sessions", json={"paper_id": paper_id})).status_code, 409)
        _, session_id = await self.ready_session()
        for query in ("", " ", "research " * 2100):
            self.assertEqual((await self.ask(session_id, query)).status_code, 422)
        self.chat_mock.assert_not_awaited()
        self.assertEqual((await self.history(session_id)).json(), [])

    async def test_provider_failure_saves_neither_message(self):
        _, session_id = await self.ready_session()
        self.chat_mock.side_effect = RuntimeError("private credentials error")
        response = await self.ask(session_id)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private credentials", response.text)
        self.assertEqual((await self.history(session_id)).json(), [])

    async def test_invalid_citation_saves_neither_message(self):
        _, session_id = await self.ready_session()
        self.chat_mock.side_effect = None
        self.chat_mock.return_value = {"answer": "Invented source [999]", "model_used": "test"}
        self.assertEqual((await self.ask(session_id)).status_code, 502)
        self.assertEqual((await self.history(session_id)).json(), [])

    async def test_save_failure_rolls_back_user_message(self):
        _, session_id = await self.ready_session()
        original = crud.add_message

        async def fail_assistant(db, session_id, role, content, **kwargs):
            if role == "assistant":
                raise SQLAlchemyError("Test write failure")
            return await original(db, session_id, role, content, **kwargs)

        with patch.object(crud, "add_message", side_effect=fail_assistant):
            response = await self.ask(session_id)
        self.assertEqual(response.status_code, 503)
        self.assertEqual((await self.history(session_id)).json(), [])

    async def test_no_sources_requires_no_gateway_call(self):
        _, session_id = await self.ready_session()
        self.index.records.clear()
        response = await self.ask(session_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"], [])
        self.assertEqual(response.json()["model_used"], "none")
        self.chat_mock.assert_not_awaited()

    async def test_deleted_paper_removes_session_and_chat(self):
        paper_id, session_id = await self.ready_session()
        await self.ask(session_id)
        self.assertEqual((await self.client.delete(f"/api/v1/papers/{paper_id}")).status_code, 200)
        self.assertEqual((await self.history(session_id)).status_code, 404)
        self.assertEqual((await self.ask(session_id)).status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
