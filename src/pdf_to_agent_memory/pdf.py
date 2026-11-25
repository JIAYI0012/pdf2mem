"""Read PDF text and produce deterministic, page-grounded chunks.

This module uses the PDF's existing text layer; it does not perform OCR or
interpret figures and tables. Page numbers are physical, one-based PDF pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader


class PDFExtractionError(Exception):
    """A PDF could not be read completely or contains no extractable text."""


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class PDFChunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    # Character offsets into the page's extracted text, end exclusive.
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class PDFDocument:
    document_id: str
    source_path: str
    metadata: dict[str, Any]
    pages: list[PDFPage]
    chunks: list[PDFChunk]
    # Blank pages are included: a scan or extraction issue may need review.
    empty_pages: list[int]


def _page_chunks(
    page: PDFPage, document_id: str, chunk_size: int, chunk_overlap: int
) -> list[PDFChunk]:
    """Split one page, preserving exact substrings and adjacent overlap."""
    text = page.text
    chunks: list[PDFChunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer a word boundary near the target size, but guarantee progress.
            boundary = text.rfind(" ", start + max(1, chunk_size * 4 // 5), end)
            if boundary > start + chunk_overlap:
                end = boundary + 1
        part = text[start:end]
        if part.strip():
            token = sha256(
                f"{document_id}:{page.page_number}:{start}:{end}:".encode("utf-8")
                + part.encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                PDFChunk(
                    chunk_id=f"p{page.page_number}-{token}",
                    text=part,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    start_offset=start,
                    end_offset=end,
                )
            )
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def extract_pdf(
    path: str | Path, *, chunk_size: int = 10000, chunk_overlap: int = 1000
) -> PDFDocument:
    """Extract PDF text, metadata and page-level chunks.

    The document ID hashes the file bytes, so moving or renaming a file does
    not change its identity. Chunk IDs also encode page, offsets and text.
    Raises PDFExtractionError for unreadable, password-protected, corrupt or
    textless PDFs, and for a page extraction error instead of silently skipping
    that page. Individual empty pages are reported in ``empty_pages``.
    """
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= chunk_overlap < chunk_size")

    pdf_path = Path(path)
    try:
        data = pdf_path.read_bytes()
    except OSError as exc:
        raise PDFExtractionError(f"Cannot read PDF {pdf_path}: {exc}") from exc
    document_id = sha256(data).hexdigest()

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFExtractionError(f"PDF is password-protected: {pdf_path}")
        raw_meta = reader.metadata or {}
        metadata = {str(key).lstrip("/"): str(value) for key, value in raw_meta.items() if value is not None}
        metadata["page_count"] = len(reader.pages)
        pages: list[PDFPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                raise PDFExtractionError(
                    f"Could not extract page {index} of {pdf_path}: {exc}"
                ) from exc
            pages.append(PDFPage(page_number=index, text=page_text))
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"Cannot parse PDF {pdf_path}: {exc}") from exc

    empty_pages = [page.page_number for page in pages if not page.text.strip()]
    if not pages or len(empty_pages) == len(pages):
        raise PDFExtractionError(
            f"No extractable text in PDF {pdf_path}; it may require OCR"
        )
    chunks = [
        chunk
        for page in pages
        for chunk in _page_chunks(page, document_id, chunk_size, chunk_overlap)
    ]
    return PDFDocument(
        document_id=document_id,
        source_path=str(pdf_path),
        metadata=metadata,
        pages=pages,
        chunks=chunks,
        empty_pages=empty_pages,
    )
