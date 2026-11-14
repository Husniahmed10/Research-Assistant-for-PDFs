"""Paper upload, background summary generation, and document management."""

import json
import logging
from datetime import datetime
from pathlib import PureWindowsPath
from uuid import UUID, uuid4

import logfire
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pinecone import Index
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from core.dependencies import get_db, get_pinecone_index
from core.exceptions import ErrorHandlingRoute
from db import crud
from db.database import AsyncSessionLocal
from db.models import Paper
from services.embedder import embed_chunks
from services.pdf_processor import process_pdf_document
from services.pinecone_client import delete_paper_vectors, upsert_chunks
from services.summarizer import generate_summary
from utils.file_utils import (
    delete_upload_file, get_file_size_kb, save_upload_file, validate_pdf,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/papers", tags=["Papers"], route_class=ErrorHandlingRoute)


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


async def process_and_index(paper_id: str, file_path: str, index: Index) -> None:
    """Extract, summarize, embed and index with independent database sessions.

    BackgroundTasks are not durable; a server restart can interrupt this work.
    """
    indexing_started = False
    stage = "extraction"
    try:
        document = await run_in_threadpool(process_pdf_document, file_path)
        async with AsyncSessionLocal() as db:
            paper = await crud.get_paper(db, paper_id)
            if paper is None or paper.status != "processing":
                return
            paper.full_text = document["full_text"]
            paper.page_count = document["page_count"]
            await db.commit()

        stage = "summary"
        summary = await generate_summary(document["full_text"])
        stage = "embedding"
        chunks = await embed_chunks(document["chunks"])
        stage = "indexing"
        indexing_started = True
        await run_in_threadpool(upsert_chunks, index, chunks, paper_id)
        stage = "finalization"
        async with AsyncSessionLocal() as db:
            async with db.begin():
                paper = await crud.get_paper(db, paper_id)
                if paper is None or paper.status != "processing":
                    raise ValueError("Paper is no longer awaiting processing")
                await crud.update_paper_summary(db, paper_id, summary)
                await crud.update_paper_status(db, paper_id, "ready")
    except Exception as exc:
        # Do not include provider errors or document contents in API responses.
        logfire.error("Paper processing failed", paper_id=paper_id, stage=stage, error_type=type(exc).__name__)
        logger.error("Paper processing failed for %s at %s (%s)", paper_id, stage, type(exc).__name__)
        if indexing_started:
            try:
                await run_in_threadpool(delete_paper_vectors, index, paper_id)
            except Exception:
                logger.error("Could not clean up vectors for failed paper %s", paper_id)
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
    index: Index = Depends(get_pinecone_index),
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

    background_tasks.add_task(process_and_index, paper_id, file_path, index)
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
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
    index: Index = Depends(get_pinecone_index),
) -> dict[str, str]:
    paper = await _require_paper(db, str(paper_id))
    if paper.status in {"uploading", "processing"}:
        raise HTTPException(status_code=409, detail="Cannot delete a paper while processing")
    try:
        await run_in_threadpool(delete_paper_vectors, index, paper.id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not delete vectors; please retry") from exc
    try:
        await run_in_threadpool(delete_upload_file, paper.file_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Could not delete the PDF; please retry") from exc
    await crud.delete_paper(db, paper.id)
    await db.commit()
    return {"message": "deleted"}
