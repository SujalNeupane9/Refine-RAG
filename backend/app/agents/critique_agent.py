import json
import logging
import re
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_MODEL, GOOGLE_API_KEY
from app.graph.state import RAGState


logger = logging.getLogger(__name__)

RETRIEVAL_CRITIQUE_PROMPT_TEMPLATE = """You are the Critique Agent for a RAG system.
Evaluate whether the retrieved context is relevant to the user query.
Return JSON only:
{{
  "score": 0.0 to 1.0,
  "critique": "short explanation",
  "refined_query": "better query if score is low, otherwise empty string"
}}

User query:
{query}

Context:
{context}
"""

ANSWER_CRITIQUE_PROMPT_TEMPLATE = """You are the Critique Agent for RefineRAG.
Check whether the answer is supported by the context.
Return JSON only:
{{
  "score": 0.0 to 1.0,
  "critique": "short explanation"
}}

User query:
{query}

Context:
{context}

Answer:
{answer}
"""


def critique_retrieval_node(state: RAGState) -> RAGState:
    logger.info(
        "Retrieval critique started: context_characters=%s iteration=%s",
        len(state.get("context", "")),
        state.get("iteration", 0),
    )
    prompt = RETRIEVAL_CRITIQUE_PROMPT_TEMPLATE.format(
        query=state["query"],
        context=state.get("context", ""),
    )
    try:
        raw_response = _normalize_content(_get_llm().invoke(prompt).content)
        payload = _extract_json_payload(raw_response)
    except Exception:
        logger.exception("Retrieval critique failed; using local fallback.")
        payload = _local_retrieval_fallback(state)

    score = _clamp_score(payload.get("score", 0.0))
    critique = str(payload.get("critique", "")).strip()
    refined_query = str(payload.get("refined_query", "")).strip() or None

    logger.info(
        "Retrieval critique completed: score=%s has_refined_query=%s",
        score,
        bool(refined_query),
    )

    return {
        **state,
        "retrieval_score": score,
        "critique": critique,
        "refined_query": refined_query,
    }


def critique_answer_node(state: RAGState) -> RAGState:
    logger.info(
        "Answer critique started: answer_characters=%s",
        len(state.get("answer", "") or ""),
    )
    prompt = ANSWER_CRITIQUE_PROMPT_TEMPLATE.format(
        query=state["query"],
        context=state.get("context", ""),
        answer=state.get("answer", "") or "",
    )
    try:
        raw_response = _normalize_content(_get_llm().invoke(prompt).content)
        payload = _extract_json_payload(raw_response)
    except Exception:
        logger.exception("Answer critique failed; using local fallback.")
        payload = _local_answer_fallback(state)

    score = _clamp_score(payload.get("score", 0.0))
    critique = str(payload.get("critique", "")).strip()

    logger.info("Answer critique completed: score=%s", score)

    return {
        **state,
        "answer_score": score,
        "critique": critique,
    }


def _get_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        logger.error("Cannot initialize Critique Agent because GOOGLE_API_KEY is missing.")
        raise ValueError("GOOGLE_API_KEY is required to initialize the Critique Agent.")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def _normalize_content(raw_response: Any) -> str:
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


def _extract_json_payload(raw_response: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    logger.warning("Critique Agent returned non-JSON output; using fallback payload.")
    return {"score": 0.0, "critique": "Critique response could not be parsed.", "refined_query": ""}


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, score))


def _local_retrieval_fallback(state: RAGState) -> Dict[str, Any]:
    context = state.get("context", "").strip()
    if not context:
        return {
            "score": 0.0,
            "critique": "No relevant context was retrieved.",
            "refined_query": state["query"],
        }

    return {
        "score": 1.0,
        "critique": "The retrieved context appears relevant to the query.",
        "refined_query": "",
    }


def _local_answer_fallback(state: RAGState) -> Dict[str, Any]:
    answer = (state.get("answer", "") or "").strip().lower()
    context = (state.get("context", "") or "").strip().lower()

    if not answer or not context:
        return {
            "score": 0.0,
            "critique": "The answer is not supported by the retrieved context.",
        }

    if "i don't have enough information" in answer or "insufficient" in answer:
        return {
            "score": 0.2,
            "critique": "The answer correctly signals limited context.",
        }

    return {
        "score": 1.0,
        "critique": "The answer appears supported by the retrieved context.",
    }
