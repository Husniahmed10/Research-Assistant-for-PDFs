"""Paper upload, background summary generation, and document management."""

import json
import logging
from datetime import datetime
from pathlib import PureWindowsPath
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from core.dependencies import get_db
from db import crud
from db.database import AsyncSessionLocal
from db.models import Paper
from services.pdf_processor import process_pdf_document
from services.summarizer import generate_summary
from utils.file_utils import (
    delete_upload_file, get_file_size_kb, save_upload_file, validate_pdf,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/papers", tags=["Papers"])


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    status: str


class PaperStatus(BaseModel):
    paper_id: str
    status: str
    page_count: int | None


class PaperInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_size_kb: float
    page_count: int | None
    status: str
    created_at: datetime


async def _require_paper(db: AsyncSession, paper_id: str) -> Paper:
    paper = await crud.get_paper(db, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


async def process_and_index(paper_id: str, file_path: str) -> None:
    """Extract/chunk and summarize using sessions independent of the request.

    Vector indexing is deferred until the Pinecone service is implemented.
    BackgroundTasks are not durable; a server restart can interrupt this work.
    """
    try:
        document = await run_in_threadpool(process_pdf_document, file_path)
        async with AsyncSessionLocal() as db:
            paper = await crud.get_paper(db, paper_id)
            if paper is None or paper.status != "processing":
                return
            paper.full_text = document["full_text"]
            paper.page_count = document["page_count"]
            await db.commit()

        summary = await generate_summary(document["full_text"])
        async with AsyncSessionLocal() as db:
            async with db.begin():
                paper = await crud.get_paper(db, paper_id)
                if paper is None or paper.status != "processing":
                    return
                await crud.update_paper_summary(db, paper_id, summary)
                await crud.update_paper_status(db, paper_id, "ready")
    except Exception:
        # Do not include provider errors or document contents in API responses.
        logger.error("Paper processing failed for %s", paper_id)
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await crud.update_paper_status(db, paper_id, "failed")
        except Exception:
            logger.error("Could not persist failed status for paper %s", paper_id)


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    file_path = None
    try:
        # Helpers use 413/415; this endpoint exposes 400 for invalid uploads.
        try:
            await validate_pdf(file)
            paper_id = str(uuid4())
            file_path = await save_upload_file(file, paper_id)
        except HTTPException as exc:
            if exc.status_code in {413, 415}:
                raise HTTPException(status_code=400, detail=exc.detail) from exc
            raise
        filename = PureWindowsPath(file.filename or "document.pdf").name
        paper = await crud.create_paper(
            db,
            filename=filename,
            file_path=file_path,
            file_size_kb=await run_in_threadpool(get_file_size_kb, file_path),
            paper_id=paper_id,
            status="processing",
        )
        response = UploadResponse(paper_id=paper.id, filename=paper.filename, status=paper.status)
        await db.commit()
    except Exception:
        await db.rollback()
        if file_path is not None:
            try:
                await run_in_threadpool(delete_upload_file, file_path)
            except (OSError, ValueError):
                logger.error("Could not clean up an unsuccessful upload")
        raise
    finally:
        await file.close()

    background_tasks.add_task(process_and_index, paper_id, file_path)
    return response


@router.get("/{paper_id}/status", response_model=PaperStatus)
async def get_paper_status(
    paper_id: UUID, db: AsyncSession = Depends(get_db)
) -> PaperStatus:
    paper = await _require_paper(db, str(paper_id))
    return PaperStatus(paper_id=paper.id, status=paper.status, page_count=paper.page_count)


@router.get("/{paper_id}/summary", response_model=dict[str, str])
async def get_paper_summary(
    paper_id: UUID, db: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    paper = await _require_paper(db, str(paper_id))
    if paper.status == "failed":
        raise HTTPException(status_code=409, detail="Paper processing failed; delete and re-upload the PDF")
    if paper.status != "ready":
        raise HTTPException(status_code=409, detail="Paper is still processing")
    if not paper.summary_json:
        raise HTTPException(status_code=409, detail="Summary is not available")
    return json.loads(paper.summary_json)


@router.get("/", response_model=list[PaperInfo])
async def list_papers(db: AsyncSession = Depends(get_db)) -> list[Paper]:
    return await crud.get_all_papers(db)


@router.delete("/{paper_id}")
async def delete_paper(
    paper_id: UUID, db: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    paper = await _require_paper(db, str(paper_id))
    if paper.status in {"uploading", "processing"}:
        raise HTTPException(status_code=409, detail="Cannot delete a paper while processing")
    try:
        await run_in_threadpool(delete_upload_file, paper.file_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Could not delete the PDF; please retry") from exc
    await crud.delete_paper(db, paper.id)
    await db.commit()
    return {"message": "deleted"}
