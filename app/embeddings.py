from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": False},
        )
    return _embeddings
