import os
import json
from dotenv import load_dotenv
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq

from retrieval.hybrid_retriever import HybridRetriever, Retrieved

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"

# CrossEncoder scores are relative, not probabilities. A threshold of -2.0
# is too strict for follow-up questions and filters out otherwise relevant
# chunks. Use a more permissive threshold.
RELEVANCE_THRESHOLD = -4.0

REFUSAL_TEXT = "The requested information is unavailable in the provided textbooks."


class ChatState(TypedDict):
    question: str
    chat_history: list[dict]
    available_textbooks: list[dict]
    standard_filter: Optional[str]
    subject_filter: Optional[str]
    retrieved_chunks: list[Retrieved]
    answer: str
    citations: list[dict]


def _get_groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def rewrite_query_node(state: ChatState) -> ChatState:
    print("🔥 rewrite_query_node called")
    if not state["chat_history"]:
        state["question"] = state["question"]
        return state

    client = _get_groq_client()
    last_turns = state["chat_history"][-4:]
    history_context = "\n".join(f"{t['role']}: {t['content']}" for t in last_turns)

    system_prompt = """Rewrite the user's latest message into a fully standalone
question or instruction that makes sense without any conversation history, by
resolving pronouns (e.g. "that", "it", "this") and implicit references using
the conversation below. The rewrite does NOT need to be phrased as a question —
if the original is an instruction like "explain in more detail", rewrite it as
a clear standalone instruction (e.g. "Explain HTML in more detail"), not a
question. If the original is already standalone, return it unchanged.
Respond with ONLY the rewritten text, nothing else — no quotes, no explanation."""

    user_content = f"Conversation so far:\n{history_context}\n\nLatest message: {state['question']}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    rewritten = response.choices[0].message.content.strip()

    # Debug: verify the model is actually rewriting follow-up questions.
    print("=" * 60)
    print("ORIGINAL :", state["question"])
    print("REWRITTEN:", rewritten)
    print("=" * 60)

    # Sanity-check the rewrite before trusting it: reject anything that looks
    # like meta-commentary/explanation rather than an actual standalone
    # question or instruction. Deliberately does NOT require a "?" ending,
    # since valid rewrites of instructions (e.g. "explain in more detail")
    # are not phrased as questions.
    if (
        rewritten
        and len(rewritten) < 300
        and "Need to rewrite" not in rewritten
        and "The user asks" not in rewritten
        and not rewritten.lower().startswith(("here", "the user", "i need", "sure,"))
    ):
        state["question"] = rewritten
    return state


def route_query_node(state: ChatState) -> ChatState:
    client = _get_groq_client()

    textbook_list = "\n".join(
        f"- standard={tb['standard']}, subject={tb['subject']} ({tb['textbook_name']})"
        for tb in state["available_textbooks"]
    )

    system_prompt = f"""You are a routing classifier for a textbook Q&A system.
Available textbooks in the knowledge base:
{textbook_list}

Given the user's question, decide which standard and subject it most likely
relates to. Respond ONLY with JSON in this exact format:
{{"standard": "<value or null>", "subject": "<value or null>", "confidence": "<high|low>"}}

Rules:
- If the question clearly matches exactly one textbook, return that standard/subject with confidence "high".
- If the question could apply to multiple textbooks, is ambiguous, or is a
  general/follow-up question (e.g. "explain that more", "why?"), return
  null for both fields with confidence "low" so the system searches everything.
- Do not guess. When unsure, prefer null over a wrong guess.
"""

    history_context = ""
    if state["chat_history"]:
        last_turns = state["chat_history"][-4:]
        history_context = "\n".join(f"{t['role']}: {t['content']}" for t in last_turns)

    user_content = f"Recent conversation:\n{history_context}\n\nCurrent question: {state['question']}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        standard = parsed.get("standard") or None
        subject = parsed.get("subject") or None
        confidence = parsed.get("confidence", "low")
        if confidence == "low":
            standard, subject = None, None
    except (json.JSONDecodeError, AttributeError):
        standard, subject = None, None

    state["standard_filter"] = standard
    state["subject_filter"] = subject
    return state


def retrieve_node(state: ChatState, retriever: HybridRetriever) -> ChatState:
    print("🔥 retrieve_node called")
    chunks = retriever.retrieve(
        query=state["question"],
        standard=state["standard_filter"],
        subject=state["subject_filter"],
    )

    print("\n========== RETRIEVAL RESULTS ==========")
    print("Query:", state["question"])

    if not chunks:
        print("No chunks retrieved.")
    else:
        for i, c in enumerate(chunks[:5], start=1):
            print(
                f"{i}. Score={c.score:.4f} | {c.textbook_name} | Page {c.page_number}"
            )

    print("=======================================\n")

    best_score = max((c.score for c in chunks), default=None)
    needs_retry = (not chunks) or (best_score is not None and best_score < RELEVANCE_THRESHOLD)

    if needs_retry and (state["standard_filter"] or state["subject_filter"]):
        unfiltered_chunks = retriever.retrieve(query=state["question"], standard=None, subject=None)
        unfiltered_best = max((c.score for c in unfiltered_chunks), default=None)
        if unfiltered_best is not None and (best_score is None or unfiltered_best > best_score):
            chunks = unfiltered_chunks

    state["retrieved_chunks"] = chunks
    return state


def generate_node(state: ChatState) -> ChatState:
    print("🔥 generate_node called", flush=True)
    chunks = state["retrieved_chunks"]

    relevant_chunks = [c for c in chunks if c.score >= RELEVANCE_THRESHOLD]
    seen = set()
    unique_chunks = []

    for chunk in relevant_chunks:
        key = (chunk.textbook_name, chunk.page_number, chunk.text)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    relevant_chunks = unique_chunks

    if not relevant_chunks:
        state["answer"] = REFUSAL_TEXT
        state["citations"] = []
        return state

    context_block = "\n\n".join(
        f"[Source {i+1}: {c.textbook_name}, Page {c.page_number}]\n{c.text}"
        for i, c in enumerate(relevant_chunks)
    )

    print("\n========== CONTEXT SENT TO LLM ==========" , flush=True)
    print(context_block, flush=True)
    print("=========================================\n", flush=True)

    system_prompt = f"""You are a helpful study assistant that answers questions
STRICTLY using the provided textbook excerpts below. Rules:
- Only use information present in the excerpts. Do not use outside knowledge.
- If the answer cannot be answered from the excerpts, reply exactly: "{REFUSAL_TEXT}"
- If the excerpts don't fully answer the question, say what's missing rather
  than filling gaps with assumptions.
- Refer to sources naturally (e.g. "According to Source 2...") if helpful,
  but do not fabricate page numbers yourself — citations are handled separately.
- Keep answers clear and appropriately detailed for a school student.
"""

    history_messages = []
    for turn in state["chat_history"][-6:]:
        history_messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = f"Textbook excerpts:\n{context_block}\n\nQuestion: {state['question']}"

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        stream=False,
    )

    answer_text = response.choices[0].message.content
    state["answer"] = answer_text

    if answer_text.strip() == REFUSAL_TEXT:
        state["citations"] = []
    else:
        state["citations"] = [
            {
                "textbook_name": c.textbook_name,
                "page_number": c.page_number,
                "snippet": c.text[:200] + ("..." if len(c.text) > 200 else ""),
            }
            for c in relevant_chunks
        ]
    return state


def build_graph(retriever: HybridRetriever):
    graph = StateGraph(ChatState)

    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("route_query", route_query_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    graph.add_node("generate", generate_node)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "route_query")
    graph.add_edge("route_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()