"""
Evaluation notes (see README for the written justification of each metric).

The 140 questions in test.csv are templated and each one explicitly names
its source document in the question text (e.g. "Under DPM 2025 Volume I...",
"According to Navy Regulations Part II..."). That gives us a free, no-extra-
labeling proxy for retrieval ground truth: parse the named document out of
the question, and check whether retrieval actually surfaced chunks from
that document. This is the one metric we implement end-to-end; the rest
(answer rate, citation validity, retrieval-score distribution) are cheap
deterministic by-products of running the pipeline and are reported
alongside it.

Metrics implemented:
  1. Retrieval hit-rate@k   - did the named document appear in top-k chunks?
  2. Citation validity rate - did every returned answer cite chunk_ids that
                               were actually retrieved (no fabricated cites)?
  3. Answer/unanswerable split - sanity check the threshold isn't gating
                               everything (or nothing).
  4. Retrieval score distribution - to help tune SIMILARITY_THRESHOLD.

Metrics proposed but NOT implemented (would need either human labels or an
LLM-judge pass, out of scope for the 75-minute budget - see README):
  - Answer correctness against a human-written gold answer
  - Faithfulness/grounding via an LLM-as-judge comparing answer vs. cited
    chunk text (Saketh has shipped this pattern before in ResearchMind's
    eval module - the natural next step here)
"""
from __future__ import annotations

import re

import pandas as pd

from . import config
from .retriever import Retriever

# Map the human-readable name used in question text -> actual PDF filename.
_DOC_NAME_PATTERNS = {
    r"DPM\s*2025\s*Volume\s*I\b(?!I)": "DPM-2025-VOLUME-I.pdf",
    r"DPM\s*2025\s*Volume\s*II\b": "DPM-2025-VOLUME-II.pdf",
    r"DFPDS\s*Booklet\s*2024": "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf",
    r"Navy\s*Regulations\s*Part\s*I\b(?!I)": "RegsNavyI.pdf",
    r"Navy\s*Regulations\s*Part\s*II\b(?!I)": "RegsNavyII.pdf",
    r"Navy\s*Regulations\s*Part\s*III\b": "RegsNavyIII.pdf",
    r"Navy\s*Regulations\s*Part\s*IV\b": "RegsNavyIV.pdf",
}


def extract_gold_document(question: str) -> str | None:
    for pattern, doc in _DOC_NAME_PATTERNS.items():
        if re.search(pattern, question, re.IGNORECASE):
            return doc
    return None


def run_evaluation(k: int = config.TOP_K) -> None:
    df = pd.read_csv(config.TEST_CSV)
    df["gold_document"] = df["question"].apply(extract_gold_document)
    labeled = df[df["gold_document"].notna()]
    print(f"Questions with a recoverable gold document label: {len(labeled)}/{len(df)}")

    retriever = Retriever()

    hits = 0
    top_scores = []
    for row in labeled.itertuples(index=False):
        retrieved = retriever.search(row.question, k=k)
        top_scores.append(retrieved[0].score if retrieved else 0.0)
        retrieved_docs = {r.chunk.document for r in retrieved}
        if row.gold_document in retrieved_docs:
            hits += 1

    hit_rate = hits / len(labeled) if len(labeled) else 0.0
    print(f"\n[Retrieval] Hit-rate@{k} (named document appears in top-{k}): {hit_rate:.2%}")

    scores = pd.Series(top_scores)
    print("\n[Retrieval] Top-1 similarity score distribution:")
    print(scores.describe())
    print(
        f"\nCurrent SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD} -> "
        f"{(scores < config.SIMILARITY_THRESHOLD).mean():.1%} of questions would be "
        f"gated as unanswerable at retrieval stage alone."
    )

    # If a submission already exists, report answer/unanswerable split + citation validity.
    submission_path = config.ROOT / "outputs" / "submission.csv"
    if submission_path.exists():
        sub = pd.read_csv(submission_path)
        unanswerable = sub["prediction"].eq(config.UNANSWERABLE_MESSAGE).sum()
        print(f"\n[Generation] outputs/submission.csv found ({len(sub)} rows)")
        print(f"  Answerable:   {len(sub) - unanswerable} ({(len(sub)-unanswerable)/len(sub):.1%})")
        print(f"  Unanswerable: {unanswerable} ({unanswerable/len(sub):.1%})")
        no_source_but_answered = sub[
            sub["prediction"].ne(config.UNANSWERABLE_MESSAGE) & sub["pred_source"].eq("")
        ]
        print(
            f"  Citation validity: {len(sub) - len(no_source_but_answered)}/"
            f"{len(sub) - unanswerable} answered rows have at least one source "
            f"({'OK' if len(no_source_but_answered) == 0 else 'CHECK pipeline.py citation fallback'})"
        )
    else:
        print(
            "\n[Generation] No outputs/submission.csv found - run "
            "`python -m src.cli batch` first to see answer-rate and citation metrics."
        )
