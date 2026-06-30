"""
Embedding + FAISS vector store. Embeddings come from Gemini's hosted
text-embedding API (no local model weights, no torch) - cosine similarity
via L2-normalized vectors + inner product index (IndexFlatIP). Exact
search, fine at ~1.7k chunks.
"""
from __future__ import annotations

import pickle
import time

import faiss
import google.generativeai as genai
import numpy as np

from . import config
from .corpus import Chunk, load_corpus

_client_configured = False


def _ensure_client():
    global _client_configured
    if not _client_configured:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Export it or put it in a .env file."
            )
        genai.configure(api_key=config.GOOGLE_API_KEY)
        _client_configured = True


def _embed(texts: list[str], task_type: str = "retrieval_document") -> np.ndarray:
    """Embed a list of texts via Gemini, batching to respect API limits.

    task_type matters for this model: "retrieval_document" for corpus
    chunks at ingest time, "retrieval_query" for the user's question at
    search time - Gemini's embedding model is trained asymmetrically for
    these two roles, so using the wrong one quietly hurts retrieval quality.
    """
    _ensure_client()
    all_vecs = []
    batch_size = config.EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                all_vecs.extend(result["embedding"])
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  embed batch failed ({e}); retrying in {wait}s...")
                time.sleep(wait)

    vecs = np.array(all_vecs, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return (vecs / norms).astype("float32")


def build_index() -> None:
    """Ingest step: load corpus -> embed -> build FAISS index -> persist."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_corpus()
    if not chunks:
        raise RuntimeError("No chunks loaded from corpus - check config.CORPUS_SOURCE")
    print(f"Loaded {len(chunks)} chunks from corpus_source={config.CORPUS_SOURCE!r}")

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks via {config.EMBEDDING_MODEL} ...")
    vecs = _embed(texts, task_type="retrieval_document")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.CHUNK_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Indexed {index.ntotal} vectors (dim={dim})")
    print(f"  -> {config.FAISS_INDEX_PATH}")
    print(f"  -> {config.CHUNK_STORE_PATH}")


def load_index() -> tuple[faiss.Index, list[Chunk]]:
    if not config.FAISS_INDEX_PATH.exists() or not config.CHUNK_STORE_PATH.exists():
        raise RuntimeError(
            "Index not found. Run ingest first: python -m src.cli ingest"
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.CHUNK_STORE_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def embed_query(query: str) -> np.ndarray:
    return _embed([query], task_type="retrieval_query")