"""
Central config. Every tunable knob lives here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ROOT = Path(__file__).resolve().parent.parent

# ---- Data ----
RAW_PDF_DIR = ROOT / "data" / "raw_pdfs"
TEST_CSV = ROOT / "data" / "test.csv"
SAMPLE_SUBMISSION_CSV = ROOT / "data" / "sample_submission.csv"

# ---- Index artifacts ----
INDEX_DIR = ROOT / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNK_STORE_PATH = INDEX_DIR / "chunks.pkl"

# ---- Chunking ----
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Always build the corpus from the PDFs
# Which corpus to embed:
#   "pdf"      -> chunk the raw PDFs ourselves (src/pdf_chunker.py) — default
#   "metadata" -> use a pre-chunked metaData.csv if you have one (not used here)
CORPUS_SOURCE = os.environ.get("CORPUS_SOURCE", "pdf")

# ---- Embedding model ----
# ---- Embedding model ----
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 100  # Gemini batch embedding API limit per request

# ---- Retrieval ----
TOP_K = int(os.environ.get("TOP_K", 5))
SIMILARITY_THRESHOLD = float(
    os.environ.get("SIMILARITY_THRESHOLD", 0.30)
)

# ---- Generation ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-1.5-flash",
)

MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.1

UNANSWERABLE_MESSAGE = (
    "The provided corpus does not contain enough information to answer this question."
)