import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.config import TOP_K
from app.graph.state import RAGState
from app.rag.vector_store import search


logger = logging.getLogger(__name__)


def retrieve_node(state: RAGState) -> RAGState:
    query = state.get("refined_query") or state["query"]
    iteration = state.get("iteration", 0) + 1
    logger.info(
        "Retriever Agent started: using_refined_query=%s top_k=%s iteration=%s",
        bool(state.get("refined_query")),
        TOP_K,
        iteration,
    )

    documents = search(query=query, top_k=TOP_K)
    context = format_docs(documents)
    sources = extract_sources(documents)
    logger.info(
        "Retriever Agent completed: documents=%s sources=%s context_characters=%s",
        len(documents),
        len(sources),
        len(context),
    )

    return {
        **state,
        "iteration": iteration,
        "documents": documents,
        "context": context,
        "sources": sources,
    }


def format_docs(documents: List[Document]) -> str:
    formatted_docs = []

    for index, document in enumerate(documents, start=1):
        source_label = _source_label(document.metadata)
        formatted_docs.append(
            f"[Source {index}: {source_label}]\n{document.page_content.strip()}"
        )

    return "\n\n".join(formatted_docs)


def extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen_sources = set()

    for document in documents:
        source = {
            "file": document.metadata.get("file"),
            "page": document.metadata.get("page"),
            "chunk_id": document.metadata.get("chunk_id"),
            "chunk_text": document.page_content.strip(),
        }
        source_key = (source["file"], source["page"], source["chunk_id"])

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        sources.append(source)

    return sources


def _source_label(metadata: Dict[str, Any]) -> str:
    file_name = metadata.get("file") or "unknown source"
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")
    details = []

    if page is not None:
        details.append(f"page {page}")

    if chunk_id is not None:
        details.append(f"chunk {chunk_id}")

    if details:
        return f"{file_name}, {', '.join(details)}"

    return file_name
