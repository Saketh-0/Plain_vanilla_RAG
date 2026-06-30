"""
Generation layer. Calls Gemini with ONLY the retrieved chunks as context,
and forces structured JSON output so citations are a parseable field
rather than something we hope the model mentions in prose.
"""
from __future__ import annotations

import json
import re

import google.generativeai as genai

from . import config
from .retriever import RetrievedChunk

_SYSTEM_PROMPT = """You are a policy-compliance assistant answering questions \
about Indian defence procurement and financial-powers regulations, using ONLY \
the context passages provided below.

Rules:
1. Answer using ONLY information present in the context passages. Do not use \
outside knowledge, do not infer beyond what is stated.
2. If the context does not contain enough information to answer confidently, \
set "answerable" to false and leave "answer" empty. Do NOT guess.
3. Every claim in your answer must be traceable to the context. List the \
chunk_id(s) you actually used in "sources".
4. Be concise and precise - quote clause numbers, authority levels, or amounts \
exactly as they appear in the context when relevant.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact \
schema:
{"answerable": true/false, "answer": "string", "sources": ["chunk_id", ...]}
"""

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


def _build_context_block(retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for r in retrieved:
        c = r.chunk
        parts.append(
            f"[chunk_id: {c.chunk_id} | document: {c.document} | "
            f"section: {c.section}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last-resort: grab the first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def generate_answer(question: str, retrieved: list[RetrievedChunk]) -> dict:
    """Returns {"answerable": bool, "answer": str, "sources": [chunk_id,...]}"""
    _ensure_client()
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=_SYSTEM_PROMPT,
        generation_config={
            "temperature": config.TEMPERATURE,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        },
    )

    context_block = _build_context_block(retrieved)
    user_prompt = f"Context passages:\n\n{context_block}\n\nQuestion: {question}"

    response = model.generate_content(user_prompt)
    raw_text = response.text or ""

    try:
        parsed = _extract_json(raw_text)
    except Exception:
        # Model didn't follow the schema - fail safe rather than crash the batch
        parsed = {"answerable": False, "answer": "", "sources": []}

    parsed.setdefault("answerable", False)
    parsed.setdefault("answer", "")
    parsed.setdefault("sources", [])
    return parsed
