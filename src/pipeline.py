"""
Orchestration layer. Two-stage unanswerable handling, deliberately:

  Stage 1 (cheap, deterministic): if the top retrieved chunk's similarity
  score is below SIMILARITY_THRESHOLD, we never even call the LLM - the
  corpus clearly has nothing relevant. This is what stops the system from
  paying for a generation call (and risking a hallucinated answer) on
  off-topic questions.

  Stage 2 (model-judged): even when retrieval looks plausible, the LLM is
  instructed to set answerable=false if the retrieved text doesn't
  actually support an answer. Retrieval similarity is a proxy for
  relevance, not proof the answer is actually contained in the text -
  this stage catches cases where the proxy is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .generator import generate_answer
from .retriever import Retriever, RetrievedChunk


@dataclass
class RagResult:
    question: str
    answerable: bool
    answer: str
    sources: list[str]
    source_documents: list[str]
    source_sections: list[str]
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0


class RagPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, question: str, k: int = config.TOP_K) -> RagResult:
        retrieved = self.retriever.search(question, k=k)
        top_score = retrieved[0].score if retrieved else 0.0

        # Stage 1: retrieval-level gate
        if not retrieved or top_score < config.SIMILARITY_THRESHOLD:
            return RagResult(
                question=question,
                answerable=False,
                answer=config.UNANSWERABLE_MESSAGE,
                sources=[],
                source_documents=[],
                source_sections=[],
                retrieved=retrieved,
                top_score=top_score,
            )

        # Stage 2: model-level gate + generation
        gen = generate_answer(question, retrieved)

        if not gen.get("answerable"):
            return RagResult(
                question=question,
                answerable=False,
                answer=config.UNANSWERABLE_MESSAGE,
                sources=[],
                source_documents=[],
                source_sections=[],
                retrieved=retrieved,
                top_score=top_score,
            )

        cited_ids = gen.get("sources", [])
        id_to_chunk = {r.chunk.chunk_id: r.chunk for r in retrieved}
        # Only trust citations that actually correspond to retrieved chunks -
        # the model cannot cite a chunk_id we never gave it.
        valid_cited = [cid for cid in cited_ids if cid in id_to_chunk]
        if not valid_cited:
            # Model gave an answer but no valid citation -> treat as ungrounded
            valid_cited = [retrieved[0].chunk.chunk_id]

        return RagResult(
            question=question,
            answerable=True,
            answer=gen.get("answer", ""),
            sources=valid_cited,
            source_documents=sorted({id_to_chunk[c].document for c in valid_cited}),
            source_sections=sorted({id_to_chunk[c].section for c in valid_cited}),
            retrieved=retrieved,
            top_score=top_score,
        )
