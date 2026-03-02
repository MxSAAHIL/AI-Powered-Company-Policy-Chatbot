from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.vector_store import load_metadata, load_vector_store


@dataclass
class RetrievalBundle:
    metadata: dict[str, object]
    vector_store: FAISS


def load_retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        metadata=load_metadata(),
        vector_store=load_vector_store(),
    )


def retrieve(query: str, top_k: int) -> list[dict[str, object]]:
    bundle = load_retrieval_bundle()
    results = bundle.vector_store.similarity_search_with_score(query, k=top_k)

    retrieved_chunks: list[dict[str, object]] = []
    for document, score in results:
        document = _coerce_document(document)
        retrieved_chunks.append(
            {
                "row_id": int(document.metadata["row_id"]),
                "chunk_id": str(document.metadata["chunk_id"]),
                "text": document.page_content,
                "snippet": str(document.metadata["snippet"]),
                "score": float(score),
                "source_column": str(document.metadata["source_column"]),
            }
        )
    return retrieved_chunks


def _coerce_document(document: Document) -> Document:
    if not isinstance(document, Document):
        raise TypeError(f"Expected LangChain Document, received: {type(document)!r}")
    return document
