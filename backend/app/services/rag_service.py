import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from app.config import MAX_RETRIEVAL_RETRIES
from app.rag.loader import load_document
from app.rag.splitter import split_documents
from app.rag.vector_store import add_documents, reset_vector_store
from app.graph.workflow import rag_graph


logger = logging.getLogger(__name__)


def ask_question(question: str) -> Dict[str, Any]:
    logger.info("RAG service received question: characters=%s", len(question))

    initial_state = {
        "query": question,
        "refined_query": None,
        "documents": [],
        "context": "",
        "answer": None,
        "critique": None,
        "retrieval_score": None,
        "answer_score": None,
        "iteration": 0,
        "max_iterations": MAX_RETRIEVAL_RETRIES,
        "answer_retry_count": 0,
        "sources": [],
    }

    final_state = rag_graph.invoke(initial_state)
    logger.info(
        "RAG service completed: answer_characters=%s sources=%s",
        len(final_state.get("answer", "") or ""),
        len(final_state.get("sources", []) or []),
    )

    return {
        "answer": final_state.get("answer", ""),
        "sources": _normalize_sources(final_state.get("sources", [])),
        "critique": final_state.get("critique"),
    }


def _normalize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "file": source.get("file"),
            "page": source.get("page"),
            "chunk_id": source.get("chunk_id"),
            "chunk_text": source.get("chunk_text"),
        }
        for source in sources
    ]


def ingest_document(file_path: str | Path) -> int:
    path = Path(file_path)
    logger.info("RAG service ingest requested: file=%s", path.name)

    documents = load_document(path)
    chunks = split_documents(documents)
    chunks_created = add_documents(chunks)

    logger.info(
        "RAG service ingest completed: file=%s documents=%s chunks=%s",
        path.name,
        len(documents),
        chunks_created,
    )
    return chunks_created


def reset_ingestion_state(raw_data_dir: str | Path) -> None:
    raw_path = Path(raw_data_dir)
    logger.info("RAG service reset requested: raw_data_dir=%s", raw_path)
    reset_vector_store()

    if raw_path.exists():
        for child in raw_path.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    logger.info("RAG service reset completed: raw_data_dir=%s", raw_path)
