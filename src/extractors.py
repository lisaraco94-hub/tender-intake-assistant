"""
extractors.py — Text extraction utilities for multiple file formats.
No domain-specific logic; all analytical logic is delegated to the AI layer.
"""
from __future__ import annotations

import io
import re
from typing import List, Tuple


# ─── Text chunking helpers ────────────────────────────────────────

def chunk_pages(pages: List[str], max_chars: int = 120_000) -> List[str]:
    """
    Merge pages into chunks that fit within the model context window.
    Each chunk contains page markers for traceability.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for i, page_text in enumerate(pages, start=1):
        marker = f"\n\n--- PAGE {i} ---\n"
        block = marker + page_text.strip()
        block_len = len(block)

        if current_len + block_len > max_chars and current:
            chunks.append("".join(current))
            current = [block]
            current_len = block_len
        else:
            current.append(block)
            current_len += block_len

    if current:
        chunks.append("".join(current))

    return chunks


def extract_raw_text(pages: List[str]) -> str:
    """Return the full document text with page markers for single-pass AI analysis."""
    parts = []
    for i, page_text in enumerate(pages, start=1):
        parts.append(f"\n\n--- PAGE {i} ---\n{page_text.strip()}")
    return "".join(parts)


def guess_title_and_date(pages: List[str]) -> Tuple[str, str]:
    """
    Lightweight heuristic to extract a document title and date from the first page.
    Used only as metadata fallback — the AI extracts the real values.
    """
    title = "Tender Document"
    date = ""

    if not pages:
        return title, date

    lines = [l.strip() for l in pages[0].splitlines() if l.strip()]
    if lines:
        title = lines[0][:150]

    date_pat = re.compile(
        r"\b(\d{1,2}[./]\d{1,2}[./]20\d{2})\b"
        r"|"
        r"\b((?:January|February|March|April|May|June|July|August|September"
        r"|October|November|December|gennaio|febbraio|marzo|aprile|maggio"
        r"|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
        r"|Januar|Februar|März|April|Mai|Juni|Juli|August|September"
        r"|Oktober|November|Dezember)\s+20\d{2})\b",
        re.IGNORECASE,
    )

    for line in lines[:15]:
        m = date_pat.search(line)
        if m:
            date = m.group(0)
            break

    return title, date


# ─── Per-format extraction ────────────────────────────────────────

_CHUNK_SIZE = 3_000  # chars per synthetic "page" for non-PDF formats


def _extract_pdf(file_bytes: bytes) -> List[str]:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return [doc[i].get_text("text") for i in range(len(doc))]


def _extract_docx(file_bytes: bytes) -> List[str]:
    from docx import Document as DocxDocument
    doc = DocxDocument(io.BytesIO(file_bytes))

    pages: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        current.append(text)
        current_len += len(text)
        if current_len >= _CHUNK_SIZE:
            pages.append("\n".join(current))
            current = []
            current_len = 0

    if current:
        pages.append("\n".join(current))

    return pages or ["(No text content found in document)"]


def _extract_xlsx(file_bytes: bytes) -> List[str]:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    pages: List[str] = []

    for sheet in wb.worksheets:
        rows: List[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            row_text = "\t".join(cells).strip()
            if row_text:
                rows.append(row_text)
        if rows:
            pages.append(f"[Sheet: {sheet.title}]\n" + "\n".join(rows))

    wb.close()
    return pages or ["(No content found in spreadsheet)"]


def _extract_text(file_bytes: bytes) -> List[str]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [text[i:i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)] or [""]


# ─── Public dispatcher ────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {"pdf", "docx", "xlsx", "xls", "txt", "csv", "tsv", "md"}


def extract_from_file(file_bytes: bytes, filename: str) -> List[str]:
    """
    Extract text from any supported file type.
    Returns a list of page/chunk strings.
    Never raises — on any error returns a single-item list with a warning message.

    Supported: .pdf, .docx, .xlsx, .xls, .txt, .csv, .tsv, .md
    """
    import zipfile

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if ext == "pdf":
            return _extract_pdf(file_bytes)

        elif ext == "docx":
            try:
                return _extract_docx(file_bytes)
            except (zipfile.BadZipFile, Exception):
                # Fallback: file may be an old .doc binary or corrupted —
                # try to salvage whatever readable text is present.
                try:
                    return _extract_text(file_bytes)
                except Exception:
                    return [f"(Could not extract text from {filename}: not a valid DOCX/ZIP file)"]

        elif ext in ("xlsx", "xls"):
            try:
                return _extract_xlsx(file_bytes)
            except (zipfile.BadZipFile, Exception):
                try:
                    return _extract_text(file_bytes)
                except Exception:
                    return [f"(Could not extract text from {filename}: not a valid spreadsheet file)"]

        elif ext in ("txt", "csv", "tsv", "md"):
            return _extract_text(file_bytes)

        else:
            try:
                return _extract_text(file_bytes)
            except Exception:
                return [f"(Could not extract text from: {filename})"]

    except Exception as exc:
        return [f"(Unexpected error reading {filename}: {exc})"]
