import sys
from pathlib import Path
import pickle
import os

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.graph import build_graph

st.set_page_config(page_title="GSSTB Textbook Assistant", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Loading textbook index...")
def load_retriever():
    return HybridRetriever()


@st.cache_resource(show_spinner=False)
def load_available_textbooks():
    bm25_path = Path(__file__).parent.parent / "data" / "chroma_db" / "bm25_index.pkl"
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    seen = {}
    for meta in data["chunk_meta"].values():
        key = (meta["standard"], meta["subject"])
        seen[key] = {
            "standard": meta["standard"],
            "subject": meta["subject"],
            "textbook_name": meta["textbook_name"],
        }
    return list(seen.values())


def stream_final_answer(question: str, context_block: str, chat_history: list[dict]):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    system_prompt = """You are a helpful study assistant that answers questions
STRICTLY using the provided textbook excerpts below. Only use information
present in the excerpts, and say clearly if something isn't covered."""

    history_messages = [{"role": t["role"], "content": t["content"]} for t in chat_history[-6:]]
    user_content = f"Textbook excerpts:\n{context_block}\n\nQuestion: {question}"

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": system_prompt}, *history_messages,
                   {"role": "user", "content": user_content}],
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


st.title("📚 GSSTB Textbook Assistant")
st.caption("Ask questions about your Gujarat State Board textbooks (Std 9-12). Answers are grounded strictly in the indexed PDFs, with page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    retriever = load_retriever()
    available_textbooks = load_available_textbooks()
except FileNotFoundError:
    st.error("No index found. Run `python ingestion/ingestion.py` first after adding PDFs to data/pdfs/.")
    st.stop()

with st.sidebar:
    st.subheader("Indexed textbooks")
    for tb in available_textbooks:
        st.write(f"• {tb['textbook_name']}")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📖 Sources ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['textbook_name']}**, Page {c['page_number']}")
                    st.caption(c["snippet"])

question = st.chat_input("Ask a question about your textbooks...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing and retrieving..."):
            graph = build_graph(retriever)
            history_for_graph = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
            ]
            result = graph.invoke({
                "question": question,
                "chat_history": history_for_graph,
                "available_textbooks": available_textbooks,
                "standard_filter": None,
                "subject_filter": None,
                "retrieved_chunks": [],
                "answer": "",
                "citations": [],
            })

        if not result["citations"]:
            st.markdown(result["answer"])
            answer_text = result["answer"]
        else:
            context_block = "\n\n".join(
                f"[Source {i+1}: {c['textbook_name']}, Page {c['page_number']}]\n{c['snippet']}"
                for i, c in enumerate(result["citations"])
            )
            # Use the rewritten question produced by the graph (if available)
            final_question = result.get("question", question)

            answer_text = st.write_stream(
                stream_final_answer(final_question, context_block, history_for_graph)
            )

        if result["citations"]:
            with st.expander(f"📖 Sources ({len(result['citations'])})"):
                for c in result["citations"]:
                    st.markdown(f"**{c['textbook_name']}**, Page {c['page_number']}")
                    st.caption(c["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "citations": result["citations"],
    })