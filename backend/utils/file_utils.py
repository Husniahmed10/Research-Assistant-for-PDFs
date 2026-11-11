"""Upload validation and storage helpers rooted in settings.UPLOAD_DIR."""

import os
import re
from pathlib import Path, PureWindowsPath

import aiofiles
from fastapi import HTTPException, UploadFile

from core.config import settings


_READ_SIZE = 1024 * 1024


async def validate_pdf(file: UploadFile) -> bool:
    """Check MIME type, actual byte size, and signature; restore stream position.

    A matching signature is only a preliminary check. The PDF processor must
    still open the document to detect corrupt, encrypted, or unreadable PDFs.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only application/pdf is supported")

    original_position = file.file.tell()
    try:
        await file.seek(0)
        if await file.read(4) != b"%PDF":
            raise HTTPException(status_code=415, detail="Invalid PDF signature")
        size = 4
        limit = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        while chunk := await file.read(_READ_SIZE):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail="PDF exceeds upload size limit")
        if size > limit:
            raise HTTPException(status_code=413, detail="PDF exceeds upload size limit")
        return True
    finally:
        await file.seek(original_position)


async def save_upload_file(file: UploadFile, paper_id: str) -> str:
    """Save a validated PDF as {paper_id}_{sanitized_name}.pdf; return its path.

    Existing files are never overwritten. The caller retains ownership of the
    upload stream, whose position is restored after saving.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", paper_id):
        raise ValueError("paper_id must contain only letters, numbers, underscores or hyphens")
    await validate_pdf(file)

    # Handle both Windows and POSIX client paths, regardless of the server OS.
    original_name = PureWindowsPath(file.filename or "document.pdf").name
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_")
    stem = stem[:100] or "document"
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{paper_id}_{stem}.pdf"

    original_position = file.file.tell()
    created = False
    try:
        await file.seek(0)
        async with aiofiles.open(destination, "xb") as output:
            created = True
            size = 0
            while chunk := await file.read(_READ_SIZE):
                size += len(chunk)
                if size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="PDF exceeds upload size limit")
                await output.write(chunk)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="Upload file already exists") from exc
    except BaseException:
        # Also remove partial files if the request is cancelled.
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        await file.seek(original_position)
    return str(destination)


def get_file_size_kb(path: str | Path) -> float:
    """Return the file size in units of 1024 bytes."""
    return os.path.getsize(path) / 1024.0


def delete_upload_file(path: str | Path) -> bool:
    """Delete a file within UPLOAD_DIR; return False if it is already absent.

    Paths outside the upload directory (including escaping symlinks) are rejected.
    Other filesystem failures propagate so callers can report/retry cleanup.
    """
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    target = Path(path).resolve()
    if target == upload_dir or not target.is_relative_to(upload_dir):
        raise ValueError("Cannot delete a path outside the upload directory")
    try:
        os.remove(target)
    except FileNotFoundError:
        return False
    return True
