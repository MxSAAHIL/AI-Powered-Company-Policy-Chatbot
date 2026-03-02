from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking import chunk_records
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATASET_PATH
from app.data_loader import detect_knowledge_column, load_dataset, prepare_records
from app.default_knowledge import get_support_records
from app.embeddings import embed_documents
from app.vector_store import build_faiss_index, save_artifacts


def main() -> None:
    df = load_dataset(str(DATASET_PATH))
    text_column, scores = detect_knowledge_column(df)
    records = prepare_records(df, text_column)
    support_records = get_support_records(start_row_id=len(records))
    all_records = records + support_records
    chunks = chunk_records(all_records, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = embed_documents([str(chunk["text"]) for chunk in chunks])
    index = build_faiss_index(embeddings)

    metadata = {
        "dataset_path": str(DATASET_PATH),
        "detected_column": text_column,
        "row_count": len(df),
        "valid_row_count": len(records),
        "support_record_count": len(support_records),
        "chunk_count": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "candidate_columns": [score.__dict__ for score in scores],
    }
    save_artifacts(chunks, index, metadata)

    print(f"Detected knowledge column: {text_column}")
    print(f"Rows loaded: {len(df)}")
    print(f"Rows retained: {len(records)}")
    print(f"Support records added: {len(support_records)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("Saved artifacts to data/processed/")


if __name__ == "__main__":
    main()
