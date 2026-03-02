from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    model = get_embedding_model()
    embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return embedding.astype("float32")

