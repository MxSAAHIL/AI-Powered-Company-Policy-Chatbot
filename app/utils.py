from __future__ import annotations

from typing import Iterable


def compact_whitespace(value: str) -> str:
    return " ".join(value.split())


def make_snippet(text: str, length: int = 180) -> str:
    cleaned = compact_whitespace(text)
    if len(cleaned) <= length:
        return cleaned
    return cleaned[: length - 3].rstrip() + "..."


def batched(values: Iterable[str], batch_size: int) -> list[list[str]]:
    batch: list[str] = []
    batches: list[list[str]] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches

