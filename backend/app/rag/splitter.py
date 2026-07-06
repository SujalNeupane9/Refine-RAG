import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    logger.info(
        "Splitting documents: count=%s chunk_size=%s chunk_overlap=%s",
        len(documents),
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": str(index),
        }

    logger.info("Created document chunks: count=%s", len(chunks))
    return chunks
