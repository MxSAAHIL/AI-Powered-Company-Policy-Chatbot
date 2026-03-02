from __future__ import annotations

import re

from app.config import SNIPPET_LENGTH
from app.utils import make_snippet

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_oversized_text(text: str, chunk_size: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()]
    if len(sentences) <= 1:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                parts.append(current.strip())
            if len(sentence) > chunk_size:
                parts.extend(
                    sentence[i : i + chunk_size].strip()
                    for i in range(0, len(sentence), chunk_size)
                    if sentence[i : i + chunk_size].strip()
                )
                current = ""
            else:
                current = sentence
    if current:
        parts.append(current.strip())
    return parts


def _paragraph_units(text: str, chunk_size: int) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_oversized_text(paragraph, chunk_size))
    return units or _split_oversized_text(text, chunk_size)


def chunk_record(record: dict[str, object], chunk_size: int, chunk_overlap: int) -> list[dict[str, object]]:
    text = str(record["source_text"])
    units = _paragraph_units(text, chunk_size)
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap}{unit}".strip()
        else:
            chunks.append(unit[:chunk_size].strip())
            current = unit[max(chunk_size - chunk_overlap, 0) :].strip()

    if current:
        chunks.append(current.strip())

    chunk_records: list[dict[str, object]] = []
    for chunk_index, chunk_text in enumerate(chunks):
        chunk_records.append(
            {
                "row_id": int(record["row_id"]),
                "chunk_id": f"row_{int(record['row_id'])}_chunk_{chunk_index}",
                "chunk_index": chunk_index,
                "text": chunk_text,
                "snippet": make_snippet(chunk_text, SNIPPET_LENGTH),
                "source_column": record["source_column"],
            }
        )
    return chunk_records


def chunk_records(records: list[dict[str, object]], chunk_size: int, chunk_overlap: int) -> list[dict[str, object]]:
    all_chunks: list[dict[str, object]] = []
    for record in records:
        all_chunks.extend(chunk_record(record, chunk_size, chunk_overlap))
    return all_chunks

