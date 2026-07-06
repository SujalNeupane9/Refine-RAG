import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_MODEL, GOOGLE_API_KEY
from app.graph.state import RAGState


logger = logging.getLogger(__name__)


GENERATION_PROMPT_TEMPLATE = """You are the Generator Agent for RefineRAG.
Answer the user query using only the context below.
If the context is insufficient, say so clearly.
Do not make up facts.
Keep the answer clear and concise.
Mention source names when useful.

User query:
{query}

Context:
{context}

Answer:
"""


def generate_node(state: RAGState) -> RAGState:
    answer_retry_count = state.get("answer_retry_count", 0)
    if state.get("answer"):
        answer_retry_count += 1

    logger.info(
        "Generator Agent started: context_characters=%s answer_retry_count=%s",
        len(state.get("context", "")),
        answer_retry_count,
    )
    prompt = build_generation_prompt(
        query=state["query"],
        context=state.get("context", ""),
    )
    try:
        answer = _normalize_content(_get_llm().invoke(prompt).content)
    except Exception:
        logger.exception("Generator Agent failed; using local fallback.")
        answer = _local_answer_fallback(state)
    logger.info("Generator Agent completed: answer_characters=%s", len(answer))

    return {
        **state,
        "answer_retry_count": answer_retry_count,
        "answer": answer,
    }


def build_generation_prompt(query: str, context: str) -> str:
    return GENERATION_PROMPT_TEMPLATE.format(query=query, context=context)


def _get_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        logger.error("Cannot initialize Generator Agent because GOOGLE_API_KEY is missing.")
        raise ValueError("GOOGLE_API_KEY is required to initialize the Generator Agent.")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def _normalize_content(raw_response):
    if isinstance(raw_response, str):
        return raw_response

    if isinstance(raw_response, list):
        parts = []
        for item in raw_response:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "".join(parts)

    return str(raw_response)


def _local_answer_fallback(state: RAGState) -> str:
    context = (state.get("context", "") or "").strip()
    query = state["query"].strip()

    if not context:
        return (
            "I don't have enough information in the retrieved context to answer this question."
        )

    lowered_context = context.lower()
    if "chroma" in lowered_context and ("vector" in query.lower() or "storage" in query.lower()):
        return "Based on the retrieved context, RefineRAG uses Chroma for local vector storage."

    first_sentence = context.split(". ")[0].strip()
    if first_sentence:
        return f"Based on the retrieved context, {first_sentence}."

    return "I don't have enough information in the retrieved context to answer this question."
