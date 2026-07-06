import logging
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import TOP_K, VECTOR_STORE_PATH
from app.rag.embeddings import get_embeddings


COLLECTION_NAME = "refinerag_documents"
logger = logging.getLogger(__name__)


def get_vector_store(embedding_function: Optional[Embeddings] = None) -> Chroma:
    persist_directory = _vector_store_path()
    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading Chroma vector store: collection=%s path=%s",
        COLLECTION_NAME,
        persist_directory,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function or get_embeddings(),
        persist_directory=str(persist_directory),
    )


def add_documents(
    documents: List[Document],
    embedding_function: Optional[Embeddings] = None,
) -> int:
    if not documents:
        logger.info("No documents provided for vector indexing.")
        return 0

    vector_store = get_vector_store(embedding_function=embedding_function)
    ids = [_document_id(document, index) for index, document in enumerate(documents)]
    vector_store.add_documents(documents=documents, ids=ids)
    logger.info("Added documents to vector store: count=%s", len(documents))

    return len(documents)


def search(
    query: str,
    top_k: int = TOP_K,
    embedding_function: Optional[Embeddings] = None,
) -> List[Document]:
    vector_store = get_vector_store(embedding_function=embedding_function)
    results = vector_store.similarity_search(query, k=top_k)
    logger.info("Vector search completed: top_k=%s results=%s", top_k, len(results))
    return results


def reset_vector_store() -> None:
    persist_directory = _vector_store_path()

    if persist_directory.exists():
        logger.info("Resetting vector store: path=%s", persist_directory)
        shutil.rmtree(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info("Vector store reset complete: path=%s", persist_directory)


def _vector_store_path() -> Path:
    return Path(VECTOR_STORE_PATH)


def _document_id(document: Document, index: int) -> str:
    source = document.metadata.get("source", "document")
    chunk_id = document.metadata.get("chunk_id")

    if chunk_id is not None:
        return f"{source}:{chunk_id}"

    return f"{source}:{index}:{uuid4()}"
