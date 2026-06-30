"""
Ingestion: chunk the raw PDFs in data/raw_pdfs/ into passages. This is the
default and only corpus source — no pre-chunked metadata file.

Using pypdf (pure Python, no compiled DLL dependencies) rather than
PyMuPDF: this machine's Application Control policy blocks unsigned
compiled extensions, which ruled out PyMuPDF despite its better text
extraction quality. pypdf is the safe tradeoff here - slightly noisier
extraction on complex layouts, zero binary dependency risk.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from . import config


@dataclass
class Chunk:
    chunk_id: str
    document: str
    section: str
    text: str
    topic: str = ""


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


def _guess_section_heading(page_text: str) -> str:
    for line in page_text.splitlines():
        line = line.strip()
        if 3 < len(line) < 120 and not line.endswith((".", ",")):
            return line
    return ""


def chunk_pdf(pdf_path: Path) -> Iterator[Chunk]:
    reader = PdfReader(str(pdf_path))
    doc_name = pdf_path.name
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        heading = _guess_section_heading(page_text)
        section_label = heading if heading else f"page {page_num}"
        for piece in _split_text(
            page_text, config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS
        ):
            yield Chunk(
                chunk_id=f"{doc_name}_p{page_num:04d}_{uuid.uuid4().hex[:6]}",
                document=doc_name,
                section=section_label,
                text=piece,
            )


def chunk_all_pdfs(pdf_dir: Path = config.RAW_PDF_DIR) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_chunks = list(chunk_pdf(pdf_path))
        all_chunks.extend(doc_chunks)
        print(f"  {pdf_path.name}: {len(doc_chunks)} chunks")
    return all_chunks