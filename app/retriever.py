from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.embeddings import embed_query
from app.vector_store import load_chunks, load_index, load_metadata


@dataclass
class RetrievalBundle:
    chunks: pd.DataFrame
    metadata: dict[str, object]
    index: object


def load_retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        chunks=load_chunks(),
        metadata=load_metadata(),
        index=load_index(),
    )


def retrieve(query: str, top_k: int) -> list[dict[str, object]]:
    bundle = load_retrieval_bundle()
    query_embedding = embed_query(query)
    distances, indices = bundle.index.search(query_embedding, top_k)

    results: list[dict[str, object]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(bundle.chunks):
            continue
        row = bundle.chunks.iloc[int(idx)]
        results.append(
            {
                "row_id": int(row["row_id"]),
                "chunk_id": str(row["chunk_id"]),
                "text": str(row["text"]),
                "snippet": str(row["snippet"]),
                "score": float(score),
                "source_column": str(row["source_column"]),
            }
        )
    return results

