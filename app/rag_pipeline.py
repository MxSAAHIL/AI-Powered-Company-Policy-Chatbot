from __future__ import annotations

from app.config import TOP_K
from app.llm import GeminiClient
from app.retriever import retrieve
from app.small_talk import match_small_talk


def answer_question(question: str, top_k: int = TOP_K) -> dict[str, object]:
    small_talk_response = match_small_talk(question)
    if small_talk_response is not None:
        return {
            "answer": small_talk_response,
            "citations": [],
            "used_context": False,
        }

    retrieved_chunks = retrieve(question, top_k=top_k)
    if not retrieved_chunks:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": [],
            "used_context": False,
        }

    client = GeminiClient()
    answer = client.generate_grounded_answer(question, retrieved_chunks)
    citations = [
        {
            "row_id": chunk["row_id"],
            "chunk_id": chunk["chunk_id"],
            "snippet": chunk["snippet"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]
    return {
        "answer": answer,
        "citations": citations,
        "used_context": True,
    }
