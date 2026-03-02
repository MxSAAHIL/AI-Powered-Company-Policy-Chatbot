from __future__ import annotations

import re

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINE_PATTERN = re.compile(r"\n{3,}")


def clean_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    value = CONTROL_CHAR_PATTERN.sub("", value)
    value = WHITESPACE_PATTERN.sub(" ", value)
    value = NEWLINE_PATTERN.sub("\n\n", value)
    return value.strip()


def is_meaningful_text(text: str, min_length: int = 20) -> bool:
    return bool(text and len(text.strip()) >= min_length)

