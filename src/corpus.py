"""
Loads the corpus from the raw PDF documents.

The PDFs are parsed, split into overlapping chunks,
and returned as Chunk objects for indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pdf_chunker import chunk_all_pdfs


@dataclass
class Chunk:
    chunk_id: str
    document: str
    section: str
    text: str
    topic: str = ""


def load_corpus() -> list[Chunk]:
    pdf_chunks = chunk_all_pdfs()

    return [
        Chunk(
            chunk_id=c.chunk_id,
            document=c.document,
            section=c.section,
            text=c.text,
            topic=c.topic,
        )
        for c in pdf_chunks
    ]