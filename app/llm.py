from __future__ import annotations

import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiClient:
    def __init__(self) -> None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY. Add it to your .env file before running the app.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = GEMINI_MODEL
        self.model = genai.GenerativeModel(self.model_name)

    def generate_grounded_answer(self, question: str, retrieved_chunks: list[dict[str, object]]) -> str:
        context = self._build_context(retrieved_chunks)
        prompt = (
            "You are a retrieval-augmented assistant.\n"
            "Answer only from the provided context.\n"
            "If the answer is not in the context, say exactly: "
            "'I don't know based on the provided context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        try:
            response = self.model.generate_content(prompt)
        except Exception as exc:  # pragma: no cover - external API
            message = str(exc)
            raise RuntimeError(
                "Gemini request failed. Check your API key and selected model. "
                "If the configured model is unavailable, try gemini-2.5-flash-lite "
                "or gemini-2.5-flash."
            ) from exc

        text = getattr(response, "text", "") or ""
        if not text.strip():
            return "I don't know based on the provided context."
        return text.strip()

    @staticmethod
    def _build_context(retrieved_chunks: list[dict[str, object]]) -> str:
        parts = []
        for chunk in retrieved_chunks:
            parts.append(
                "\n".join(
                    [
                        f"row_id: {chunk['row_id']}",
                        f"chunk_id: {chunk['chunk_id']}",
                        f"text: {chunk['text']}",
                    ]
                )
            )
        return "\n\n".join(parts)

