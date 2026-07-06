import logging
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
logger = logging.getLogger(__name__)


def load_document(file_path: str | Path) -> List[Document]:
    path = Path(file_path)
    extension = path.suffix.lower()
    logger.info("Loading document: file=%s extension=%s", path.name, extension)

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        logger.warning("Unsupported document type: file=%s extension=%s", path.name, extension)
        raise ValueError(f"Unsupported file type '{extension}'. Use one of: {supported}.")

    if extension == ".pdf":
        return _load_pdf(path)

    return [_load_text_file(path)]


def load_documents(file_paths: Iterable[str | Path]) -> List[Document]:
    documents: List[Document] = []

    for file_path in file_paths:
        documents.extend(load_document(file_path))

    logger.info("Loaded documents: count=%s", len(documents))
    return documents


def _load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    documents: List[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "file": path.name,
                    "source": str(path),
                    "page": page_index + 1,
                },
            )
        )

    logger.info("Loaded PDF pages: file=%s pages_with_text=%s", path.name, len(documents))
    return documents


def _load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    logger.info("Loaded text document: file=%s characters=%s", path.name, len(text))

    return Document(
        page_content=text,
        metadata={
            "file": path.name,
            "source": str(path),
        },
    )
