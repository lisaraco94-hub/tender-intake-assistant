"""
extractors.py — Generic text extraction utilities.
No domain-specific keywords. All analytical logic is delegated to the AI layer.
"""
from __future__ import annotations
import re
from typing import List, Tuple


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
