"""
Retrieval layer. Pure search - no LLM, no business logic about
answerability. That decision is made one layer up (pipeline.py) so it's
easy to test and tune in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .corpus import Chunk
from .embed_store import embed_query, load_index


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self):
        self.index, self.chunks = load_index()

    def search(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        qvec = embed_query(query)
        scores, indices = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results
