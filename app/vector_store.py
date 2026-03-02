from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import FAISS_DIR, METADATA_PATH, ensure_data_directories
from app.embeddings import get_embeddings


def build_documents(chunks: list[dict[str, object]]) -> list[Document]:
    documents: list[Document] = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=str(chunk["text"]),
                metadata={
                    "row_id": int(chunk["row_id"]),
                    "chunk_id": str(chunk["chunk_id"]),
                    "chunk_index": int(chunk["chunk_index"]),
                    "snippet": str(chunk["snippet"]),
                    "source_column": str(chunk["source_column"]),
                },
            )
        )
    return documents


def build_faiss_vector_store(chunks: list[dict[str, object]]) -> FAISS:
    documents = build_documents(chunks)
    if not documents:
        raise ValueError("Cannot build vector store with empty chunks.")
    return FAISS.from_documents(documents, get_embeddings())


def save_artifacts(vector_store: FAISS, metadata: dict[str, object]) -> None:
    ensure_data_directories()
    vector_store.save_local(str(FAISS_DIR))
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_metadata(path: Path = METADATA_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_vector_store(path: Path = FAISS_DIR) -> FAISS:
    index_path = path / "index.faiss"
    store_path = path / "index.pkl"
    if not index_path.exists() or not store_path.exists():
        raise FileNotFoundError(f"Missing LangChain FAISS artifacts in: {path}")
    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
