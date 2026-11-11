"""Run from backend: python -m utils.test_crud

Uses an isolated in-memory database, never the configured application database.
"""

import asyncio
import json
import os

# Set before importing settings/database; changes affect this test process only.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DEBUG"] = "false"

from sqlalchemy import func, select

from db import crud
from db.database import AsyncSessionLocal, engine, init_db
from db.models import ChatSession, Message, Paper


async def main() -> None:
    try:
        await init_db()
        async with AsyncSessionLocal() as db:
            async with db.begin():
                paper = await crud.create_paper(
                    db, "dummy.pdf", "uploads/dummy.pdf", 1.5
                )
                paper_id = paper.id
                assert paper.status == "uploading"
                assert await crud.get_paper(db, paper_id) is paper
                assert [p.id for p in await crud.get_all_papers(db)] == [paper_id]
                await crud.update_paper_status(db, paper_id, "ready", page_count=2)
                await crud.update_paper_summary(db, paper_id, {"conclusion": "Dummy"})
                session = await crud.create_session(db, paper_id, "Dummy chat")
                session_id = session.id
                user = await crud.add_message(db, session_id, "user", "Findings?")
                assistant = await crud.add_message(
                    db, session_id, "assistant", "Dummy findings.",
                    retrieved_chunks=[{"page": 2, "text": "Dummy findings."}],
                    model_used="test-model", latency_ms=10,
                )
                assert user.id != assistant.id

        # A new session verifies that writes were actually committed.
        async with AsyncSessionLocal() as db:
            paper = await crud.get_paper(db, paper_id)
            assert paper.page_count == 2 and paper.status == "ready"
            assert json.loads(paper.summary_json) == {"conclusion": "Dummy"}
            sessions = await crud.get_sessions_for_paper(db, paper_id)
            assert [s.id for s in sessions] == [session_id]
            messages = await crud.get_messages_for_session(db, session_id)
            assert [m.role for m in messages] == ["user", "assistant"]
            assert messages[0].retrieved_chunks == []
            assert messages[1].retrieved_chunks[0]["page"] == 2
            assert await crud.get_paper(db, "missing") is None
            assert await crud.update_paper_status(db, "missing", "failed") is None
            assert await crud.update_paper_summary(db, "missing", {}) is None
            assert await crud.get_sessions_for_paper(db, "missing") == []
            assert await crud.get_messages_for_session(db, "missing") == []
            assert not await crud.delete_paper(db, "missing")
            for operation in (
                crud.create_session(db, "missing"),
                crud.add_message(db, "missing", "user", "Hello"),
                crud.update_paper_summary(db, paper_id, "[]"),
                crud.update_paper_status(db, paper_id, "invalid"),
                crud.add_message(db, session_id, "invalid", "Hello"),
            ):
                try:
                    await operation
                except ValueError:
                    pass
                else:
                    raise AssertionError("Expected invalid input to be rejected")
            assert await crud.delete_paper(db, paper_id)
            await db.commit()

        async with AsyncSessionLocal() as db:
            for model in (Paper, ChatSession, Message):
                assert await db.scalar(select(func.count()).select_from(model)) == 0
            # An uncommitted insertion must be rolled back.
            await crud.create_paper(db, "rollback.pdf", "uploads/rollback.pdf", 1)
            await db.rollback()
            assert await crud.get_all_papers(db) == []

        print("PASS: all 10 CRUD functions, persistence, validation, cascade deletion, rollback")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
