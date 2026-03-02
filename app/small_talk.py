from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text.lower()).strip()


def match_small_talk(text: str) -> str | None:
    normalized = _normalize(text)
    if not normalized:
        return None

    greeting_phrases = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }
    thanks_phrases = {
        "thanks",
        "thank you",
        "thankyou",
        "many thanks",
        "appreciate it",
    }
    farewell_phrases = {
        "bye",
        "goodbye",
        "see you",
        "take care",
    }

    if normalized in greeting_phrases:
        return "Hello. How can I help you with the HR policy document today?"
    if normalized in thanks_phrases:
        return "You're welcome. Let me know if you want anything else from the HR policy."
    if normalized in farewell_phrases:
        return "Goodbye. If you need more help with the HR policy, feel free to return."
    return None

