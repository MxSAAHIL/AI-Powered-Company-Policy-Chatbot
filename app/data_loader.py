from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from app.preprocess import clean_text, is_meaningful_text


@dataclass
class ColumnScore:
    name: str
    score: float
    non_null_ratio: float
    avg_length: float
    multi_word_ratio: float
    unique_ratio: float


def load_csv(path: str | None = None) -> pd.DataFrame:
    csv_path = path
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - runtime fallback
            last_error = exc
    raise RuntimeError(f"Failed to read CSV file: {csv_path}") from last_error


def load_pdf(path: str | None = None) -> pd.DataFrame:
    pdf_path = Path(path) if path is not None else None
    if pdf_path is None or not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(pdf_path))
    rows: list[dict[str, object]] = []
    for page_index, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")
        rows.append(
            {
                "page_number": page_index + 1,
                "page_text": text,
            }
        )
    return pd.DataFrame(rows)


def load_dataset(path: str | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else None
    if dataset_path is None:
        raise ValueError("Dataset path is required.")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return load_csv(str(dataset_path))
    if suffix == ".pdf":
        return load_pdf(str(dataset_path))
    raise ValueError(f"Unsupported dataset type: {dataset_path.suffix}")


def _is_likely_noise_column(column_name: str) -> bool:
    lowered = column_name.lower()
    noise_terms = [
        "id",
        "index",
        "label",
        "score",
        "class",
        "target",
        "date",
        "time",
        "url",
        "link",
        "source",
        "path",
        "file",
        "name",
        "title",
    ]
    return any(term == lowered or lowered.endswith(f"_{term}") for term in noise_terms)


def score_text_columns(df: pd.DataFrame) -> list[ColumnScore]:
    scores: list[ColumnScore] = []
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna().astype(str).map(clean_text)
        if non_null.empty:
            continue

        lengths = non_null.str.len()
        word_counts = non_null.str.split().str.len()
        non_null_ratio = float(series.notna().mean())
        avg_length = float(lengths.mean())
        multi_word_ratio = float((word_counts >= 5).mean())
        unique_ratio = float(non_null.nunique() / max(len(non_null), 1))
        penalty = 0.35 if _is_likely_noise_column(column) else 0.0
        score = (
            non_null_ratio * 0.25
            + min(avg_length / 500.0, 1.0) * 0.4
            + multi_word_ratio * 0.25
            + unique_ratio * 0.1
            - penalty
        )
        scores.append(
            ColumnScore(
                name=column,
                score=score,
                non_null_ratio=non_null_ratio,
                avg_length=avg_length,
                multi_word_ratio=multi_word_ratio,
                unique_ratio=unique_ratio,
            )
        )

    scores.sort(key=lambda item: item.score, reverse=True)
    return scores


def detect_knowledge_column(df: pd.DataFrame) -> tuple[str, list[ColumnScore]]:
    scores = score_text_columns(df)
    if not scores:
        raise ValueError("No text-like columns found in the dataset.")
    return scores[0].name, scores


def prepare_records(df: pd.DataFrame, text_column: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, value in enumerate(df[text_column].tolist()):
        cleaned = clean_text(value)
        if not is_meaningful_text(cleaned):
            continue
        records.append(
            {
                "row_id": row_id,
                "source_text": cleaned,
                "source_column": text_column,
            }
        )
    return records


def inspect_dataset(path: str) -> dict[str, Any]:
    df = load_dataset(path)
    detected_column, scores = detect_knowledge_column(df)
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "detected_column": detected_column,
        "scores": [score.__dict__ for score in scores],
        "sample_rows": prepare_records(df, detected_column)[:3],
    }
