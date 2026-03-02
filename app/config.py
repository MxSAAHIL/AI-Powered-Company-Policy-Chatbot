from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FAISS_DIR = PROCESSED_DATA_DIR / "faiss_index"

DATASET_PATH = Path(os.getenv("DATASET_PATH", str(RAW_DATA_DIR / "dataset.csv")))
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.parquet"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.json"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SNIPPET_LENGTH = int(os.getenv("SNIPPET_LENGTH", "180"))


def ensure_data_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
