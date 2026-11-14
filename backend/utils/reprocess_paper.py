"""Retry one failed upload without creating another paper or changing its file.

Run from backend: python -m utils.reprocess_paper PAPER_UUID
This calls the configured summary, embedding and vector providers.
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

import logfire
from sqlalchemy import update
from starlette.concurrency import run_in_threadpool

from api.routes.papers import process_and_index
from db import crud
from db.database import AsyncSessionLocal, engine
from db.models import Paper
from services.pinecone_client import init_pinecone_index


async def reprocess_paper(paper_id: str) -> dict:
    paper_id = str(UUID(paper_id))
    async with AsyncSessionLocal() as db:
        paper = await crud.get_paper(db, paper_id)
        if paper is None:
            raise ValueError("Paper not found")
        if paper.status != "failed":
            raise ValueError("Only failed papers can be retried; ready or processing papers are unchanged")
        if not Path(paper.file_path).is_file():
            raise ValueError("The uploaded file is missing; upload it again")
    # Check remote availability before changing a failed paper's status.
    index = await run_in_threadpool(init_pinecone_index)
    async with AsyncSessionLocal() as db:
        paper = await crud.get_paper(db, paper_id)
        if paper is None or paper.status != "failed":
            raise ValueError("Paper is no longer awaiting a retry")
        file_path = paper.file_path
        claimed = await db.execute(update(Paper).where(
            Paper.id == paper_id, Paper.status == "failed",
        ).values(status="processing"))
        if claimed.rowcount != 1:
            await db.rollback()
            raise ValueError("Another process already claimed this paper")
        await db.commit()
    await process_and_index(paper_id, file_path, index)
    async with AsyncSessionLocal() as db:
        paper = await crud.get_paper(db, paper_id)
        if paper is None or paper.status != "ready":
            raise RuntimeError("Paper could not be processed; check the logged stage and error type")
        return {"paper_id": paper.id, "filename": paper.filename, "status": paper.status,
                "page_count": paper.page_count}


async def _main(paper_id: str) -> None:
    try:
        print(json.dumps(await reprocess_paper(paper_id)))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_id", type=UUID)
    args = parser.parse_args()
    logfire.configure(send_to_logfire=False, console=False)
    engine.echo = False
    asyncio.run(_main(str(args.paper_id)))
