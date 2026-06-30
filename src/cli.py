"""
CLI entry point.

  python -m src.cli ingest
  python -m src.cli ask "What is the financial limit for ..."
  python -m src.cli batch                 # produces outputs/submission.csv
  python -m src.cli evaluate               # prints retrieval + grounding metrics
"""
from __future__ import annotations

import argparse
import sys


def cmd_ingest(_args):
    from .embed_store import build_index

    build_index()


def cmd_ask(args):
    from .pipeline import RagPipeline

    pipeline = RagPipeline()
    result = pipeline.answer(args.question, k=args.k)

    print(f"\nQuestion: {result.question}")
    print(f"Top retrieval score: {result.top_score:.3f}")
    print(f"Answerable: {result.answerable}")
    print(f"\nAnswer:\n{result.answer}")
    if result.answerable:
        print(f"\nSources (document): {', '.join(result.source_documents)}")
        print(f"Sources (section):  {', '.join(result.source_sections)}")
        print(f"Sources (chunk_id): {', '.join(result.sources)}")


def cmd_batch(args):
    from .batch_predict import run_batch

    run_batch(limit=args.limit)


def cmd_evaluate(_args):
    from .evaluate import run_evaluation

    run_evaluation()


def main():
    parser = argparse.ArgumentParser(description="Plain-vanilla RAG over the DPM/RegsNavy corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Embed the corpus and build the FAISS index")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--k", type=int, default=5)

    p_batch = sub.add_parser("batch", help="Run all questions in data/test.csv -> outputs/submission.csv")
    p_batch.add_argument("--limit", type=int, default=None, help="For quick smoke tests")

    sub.add_parser("evaluate", help="Print retrieval/grounding/unanswerable metrics")

    args = parser.parse_args()
    {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "batch": cmd_batch,
        "evaluate": cmd_evaluate,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
