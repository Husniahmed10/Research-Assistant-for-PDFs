"""Run from backend: python -m utils.test_utils.

Uses a temporary upload directory and never calls a model provider.
"""

import asyncio
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from core.config import settings
from utils.file_utils import (
    delete_upload_file, get_file_size_kb, save_upload_file, validate_pdf,
)
from utils.text_utils import (
    clean_text, count_tokens, extract_section_headers, truncate_to_tokens,
)


def upload(data: bytes, name: str = "paper.pdf", mime: str = "application/pdf") -> UploadFile:
    return UploadFile(
        file=BytesIO(data), filename=name, headers=Headers({"content-type": mime})
    )


async def check_files() -> None:
    with TemporaryDirectory() as directory:
        with patch.object(settings, "UPLOAD_DIR", str(Path(directory) / "uploads")), \
                patch.object(settings, "MAX_FILE_SIZE_MB", 1):
            data = b"%PDF-1.7\nDummy test bytes\n%%EOF"
            file = upload(data, r"..\..\unsafe name.pdf")
            try:
                await file.seek(5)
                assert await validate_pdf(file)
                assert file.file.tell() == 5
                saved = Path(await save_upload_file(file, "paper-123"))
                assert file.file.tell() == 5
                assert saved.name == "paper-123_unsafe_name.pdf"
                assert saved.parent == Path(settings.UPLOAD_DIR).resolve()
                assert saved.read_bytes() == data
                assert get_file_size_kb(saved) == len(data) / 1024
                try:
                    await save_upload_file(file, "paper-123")
                except HTTPException as exc:
                    assert exc.status_code == 409
                else:
                    raise AssertionError("Existing upload was overwritten")
                assert saved.read_bytes() == data
                try:
                    await save_upload_file(file, "../escape")
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unsafe paper ID was accepted")
                assert delete_upload_file(saved)
                assert not delete_upload_file(saved)
            finally:
                await file.close()

            for data, mime, status in (
                (b"%PDF-1.7", "text/plain", 415),
                (b"not a PDF", "application/pdf", 415),
                (b"", "application/pdf", 415),
                (b"%PDF" + b"x" * (1024 * 1024), "application/pdf", 413),
            ):
                file = upload(data, mime=mime)
                try:
                    await file.seek(1 if data else 0)
                    position = file.file.tell()
                    try:
                        await validate_pdf(file)
                    except HTTPException as exc:
                        assert exc.status_code == status
                    else:
                        raise AssertionError("Invalid upload was accepted")
                    assert file.file.tell() == position
                finally:
                    await file.close()

            outside = Path(directory) / "keep.txt"
            outside.write_text("keep", encoding="utf-8")
            try:
                delete_upload_file(outside)
            except ValueError:
                pass
            else:
                raise AssertionError("Deletion outside uploads was accepted")
            assert outside.read_text(encoding="utf-8") == "keep"


def check_text() -> None:
    assert clean_text("  A\x00\tB\n\nﬁndings\u00a0Ｃ \x07 ") == "A B findings C"
    assert clean_text("") == ""
    assert count_tokens("") == 0
    assert count_tokens("<|endoftext|>") > 0
    text = "Research findings: café, 日本語, 🌍 and experiments."
    assert truncate_to_tokens(text, count_tokens(text)) == text
    for budget in range(count_tokens(text) + 1):
        prefix = truncate_to_tokens(text, budget)
        assert text.startswith(prefix)
        assert count_tokens(prefix) <= budget
        assert "\ufffd" not in prefix
    try:
        truncate_to_tokens(text, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative token budget was accepted")
    assert extract_section_headers(
        "Abstract\nProse mentioning Results is not a heading.\n"
        "1. INTRODUCTION\r\n2.1 Methodology:\nIII. Results\n4 Conclusion\n"
    ) == ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"]
    assert extract_section_headers("The introduction discusses results.") == []


async def main() -> None:
    await check_files()
    check_text()
    print("PASS: upload validation/storage/deletion, stream positions, text cleanup, tokens, headings")


if __name__ == "__main__":
    asyncio.run(main())
