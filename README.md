# Plain-Vanilla RAG — Defence Procurement Policy QA

A grounded question-answering system over 7 Indian defence procurement /
financial-powers regulation PDFs (DPM 2025 Vol I-II, DFPDS Booklet 2024,
Navy Regulations Parts I-IV), built for the Kaggle "Defence RAG Procurement
Policy Reasoning Challenge."

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add your Gemini key (free): https://aistudio.google.com/app/apikey
export $(cat .env | xargs)  # or use python-dotenv / your shell's preferred method

python -m src.cli ingest                  # build the FAISS index (~1-2 min on CPU)
python -m src.cli ask "What authority level handles procurement approval under DPM 2025 Volume I?"
python -m src.cli batch                   # answers all of data/test.csv -> outputs/submission.csv
python -m src.cli evaluate                # retrieval/grounding metrics
```

## Pipeline & key choices

**Corpus source.** The competition ships `data/metaData.csv` — 1,668
chunks, already page-bounded and labeled with `document`, `section`, and
`topic`. I use this directly as the corpus rather than re-chunking the raw
PDFs myself (`data/raw_pdfs/`). Reasoning: it's already solved the messy
parts of PDF extraction (multi-column layout, headers/footers, tables) that
a generic chunker won't handle well on dense regulatory text in the time
available, and using it avoids introducing chunk-boundary noise the
organizers didn't intend. A from-scratch alternative is included anyway
(`src/pdf_chunker.py`, fixed 1200-char chunks / 200-char overlap) and can be
swapped in via `CORPUS_SOURCE=pdf` — useful if you want to show you can do
both, or if a different PDF corpus is substituted later.

**Embedding model.** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim,
runs fast on CPU, no API cost. At ~1.7k chunks an exact index is cheap, so
embedding quality (not index approximation) is the bottleneck; MiniLM is a
reasonable default but `bge-small-en-v1.5` is a likely upgrade if retrieval
metrics look weak (config.EMBEDDING_MODEL is a one-line swap).

**Vector store.** FAISS `IndexFlatIP` (exact inner-product search) over
L2-normalized vectors, i.e. cosine similarity. No need for IVF/HNSW
approximate search at this corpus size — exactness is free here.

**Retrieval.** Top-k=5 (`config.TOP_K`). Chosen over a smaller k because
several question types in the brief require combining information across
documents/sections, and k=5 gives the generator enough surface area without
diluting the prompt with irrelevant chunks at this corpus size.

**Generation.** Gemini (`gemini-1.5-flash` by default, configurable). The
system prompt forces strict JSON output —
`{"answerable": bool, "answer": str, "sources": [chunk_id, ...]}` — so
citations are a structured field we parse, not something we hope appears
in prose. The pipeline cross-checks every returned `chunk_id` against what
was actually retrieved; the model cannot cite something it was never shown.

**Unanswerable handling — two stages, deliberately:**
1. *Retrieval-level gate* (cheap, deterministic): if the top chunk's cosine
   similarity is below `SIMILARITY_THRESHOLD` (default 0.30, tune against
   `evaluate`'s score-distribution output), we never call the LLM. This is
   what stops the system from generating a plausible-sounding hallucination
   for an off-topic or out-of-corpus question.
2. *Model-level gate*: even when retrieval looks plausible, the LLM is
   instructed to set `answerable: false` if the retrieved text doesn't
   actually support an answer. Retrieval score is a relevance proxy, not
   proof the answer is contained in the text — stage 2 catches the cases
   where the proxy is wrong (e.g. topically similar but answer-empty
   passages).

## Evaluation notes

The 140 questions in `data/test.csv` are templated and each one explicitly
names its source document in the question text (e.g. *"Under DPM 2025
Volume I, ..."*, *"According to Navy Regulations Part II, ..."*). That's a
free, no-extra-labeling proxy for retrieval ground truth, so it's the one
metric I implement end-to-end:

| Metric | What it measures | Status |
|---|---|---|
| **Retrieval hit-rate@k** | Does the named document appear in the top-k retrieved chunks? | **Implemented** — `src/evaluate.py` |
| **Retrieval score distribution** | Top-1 cosine similarity across all questions, used to sanity-check/tune `SIMILARITY_THRESHOLD` | **Implemented** |
| **Citation validity rate** | Of answered (non-unanswerable) predictions, what fraction have at least one cited source? | **Implemented** (post-batch) |
| **Answer/unanswerable split** | Sanity check the threshold isn't gating everything or nothing | **Implemented** (post-batch) |
| Answer correctness vs. gold answer | Is the generated answer actually right | **Not implemented** — `test.csv` has no gold answer column, only a generic `contexts` field that's templated/identical across most rows and not usable as ground truth |
| Faithfulness / grounding (LLM-as-judge) | Does the answer's content actually match the cited chunk text, not just cite a plausible chunk_id | **Not implemented** — out of scope for the time budget, but the natural next step (precedent: I've shipped this exact pattern — an LLM-judge module — in a prior agentic RAG project) |

Run `python -m src.cli evaluate` after `ingest` (and optionally after
`batch`) to see all four implemented metrics.

## What I'd improve with more time

- **Hybrid retrieval**: add BM25 alongside dense retrieval and combine via
  reciprocal rank fusion — dense embeddings alone tend to underperform on
  exact clause-number / amount lookups, which this regulatory corpus has a
  lot of.
- **LLM-as-judge faithfulness scoring** against the cited chunk text, to
  catch cases where the citation is plausible but the answer drifts from
  what the chunk actually says.
- **Re-ranking**: a cross-encoder re-rank pass on the top-20 before cutting
  to top-5, since bi-encoder retrieval at this corpus size is noisy for
  multi-document combination questions.
- **Per-document threshold calibration** instead of one global similarity
  cutoff — Navy Regs and DPM volumes likely have different embedding-score
  baselines given very different writing styles.
- **Caching embedded queries** for repeated/near-duplicate questions in
  `test.csv` (several templates repeat structure across documents).

## Repo layout

```
src/
  config.py         # all tunables in one place
  corpus.py         # loads chunks from metaData.csv or PDFs
  pdf_chunker.py     # fallback PDF->chunk pipeline (not used by default)
  embed_store.py      # embedding + FAISS build/load/search
  retriever.py         # top-k search wrapper
  generator.py          # Gemini call, strict JSON+citation prompt
  pipeline.py            # ties retrieval gate + generation gate together
  batch_predict.py        # test.csv -> outputs/submission.csv
  evaluate.py               # retrieval hit-rate, score dist, citation validity
  cli.py                      # ingest / ask / batch / evaluate subcommands
data/                # metaData.csv, test.csv, sample_submission.csv, raw_pdfs/
index/               # FAISS index + chunk pickle (built by `ingest`)
outputs/             # submission.csv (built by `batch`)
```
