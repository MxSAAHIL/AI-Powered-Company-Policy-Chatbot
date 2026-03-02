from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiClient:
    def __init__(self) -> None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY. Add it to your .env file before running the app.")
        self.model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

    def generate_grounded_answer(self, question: str, retrieved_chunks: list[dict[str, object]]) -> str:
        context = self._build_context(retrieved_chunks)
        messages = [
            SystemMessage(
                content=(
                    "You are a retrieval-augmented assistant. "
                    "Answer only from the provided context. "
                    "If the answer is not in the context, say exactly: "
                    "'I don't know based on the provided context.'"
                )
            ),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                )
            ),
        ]
        try:
            response = self.model.invoke(messages)
        except Exception as exc:  # pragma: no cover - external API
            raise RuntimeError(
                "Gemini request failed through LangChain. Check your API key and selected model. "
                "If the configured model is unavailable, try gemini-2.5-flash-lite or gemini-2.5-flash."
            ) from exc

        text = getattr(response, "content", "") or ""
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        if not str(text).strip():
            return "I don't know based on the provided context."
        return str(text).strip()

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
