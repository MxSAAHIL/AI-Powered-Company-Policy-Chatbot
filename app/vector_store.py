from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from app.config import CHUNKS_PATH, FAISS_INDEX_PATH, METADATA_PATH, ensure_data_directories


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if embeddings.size == 0:
        raise ValueError("Cannot build FAISS index with empty embeddings.")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_artifacts(
    chunks: list[dict[str, object]],
    index: faiss.IndexFlatL2,
    metadata: dict[str, object],
) -> None:
    ensure_data_directories()
    pd.DataFrame(chunks).to_parquet(CHUNKS_PATH, index=False)
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def load_chunks(path: Path = CHUNKS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing chunk metadata file: {path}")
    return pd.read_parquet(path)


def load_metadata(path: Path = METADATA_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_index(path: Path = FAISS_INDEX_PATH) -> faiss.IndexFlatL2:
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index file: {path}")
    return faiss.read_index(str(path))

