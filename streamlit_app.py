from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import FAISS_DIR, GEMINI_API_KEY, TOP_K
from app.rag_pipeline import answer_question

st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon=":speech_balloon:",
    layout="wide",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            max-width: 860px;
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .title-wrap {
            text-align: center;
            margin-bottom: 1rem;
        }
        .title-wrap h1 {
            margin-bottom: 0.25rem;
            color: #11203b;
        }
        .title-wrap p {
            margin: 0;
            color: #475569;
        }
        .citation-card {
            padding: 0.85rem 0.95rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 14px;
            background: #ffffff;
            margin-bottom: 0.7rem;
        }
        .citation-meta {
            color: #475569;
            font-size: 0.86rem;
            margin-bottom: 0.45rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def artifacts_ready() -> bool:
    return (Path(FAISS_DIR) / "index.faiss").exists() and (Path(FAISS_DIR) / "index.pkl").exists()


def render_citations(citations: list[dict[str, object]]) -> None:
    if not citations:
        return
    with st.expander("Show sources", expanded=False):
        for citation in citations:
            st.markdown(
                (
                    "<div class='citation-card'>"
                    f"<div class='citation-meta'>row_id={citation['row_id']} | "
                    f"chunk_id={citation['chunk_id']}</div>"
                    f"<div>{citation['snippet']}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


st.markdown(
    """
    <div class="title-wrap">
        <h1>Financial RAG Chatbot</h1>
        <p>Ask a question and get an answer grounded only in the dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_citations(message.get("citations", []))

prompt = st.chat_input("Ask a question about the dataset")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not artifacts_ready():
            answer = "LangChain index files are missing. Run `python scripts/build_index.py` first."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "citations": []})
        elif not GEMINI_API_KEY:
            answer = "Missing GEMINI_API_KEY. Add it to `.env` and rerun the app."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "citations": []})
        else:
            try:
                with st.spinner("Thinking..."):
                    result = answer_question(prompt, top_k=TOP_K)
                st.markdown(result["answer"])
                render_citations(result["citations"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result["citations"],
                    }
                )
            except Exception as exc:
                st.error(str(exc))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": str(exc),
                        "citations": [],
                    }
                )
