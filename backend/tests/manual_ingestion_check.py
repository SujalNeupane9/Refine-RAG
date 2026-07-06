import logging
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SAMPLE_DOCUMENT = BACKEND_ROOT / "data" / "raw" / "sample_refinerag.md"
MANUAL_VECTOR_STORE = PROJECT_ROOT / ".manual_vector_store"

os.environ.setdefault("VECTOR_STORE_PATH", str(MANUAL_VECTOR_STORE))
sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.loader import load_document  # noqa: E402
from app.rag.splitter import split_documents  # noqa: E402
from app.rag.vector_store import add_documents, reset_vector_store, search  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


class LocalHashEmbeddings(Embeddings):
    dimensions = 64

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions

        for token in text.lower().split():
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            vector[index] += 1.0

        length = sum(value * value for value in vector) ** 0.5
        if not length:
            return vector

        return [value / length for value in vector]


def main() -> None:
    query = "What does the Critique Agent check?"
    embeddings = LocalHashEmbeddings()

    reset_vector_store()
    documents = load_document(SAMPLE_DOCUMENT)
    chunks = split_documents(documents)
    indexed_count = add_documents(chunks, embedding_function=embeddings)
    results = search(query, top_k=2, embedding_function=embeddings)

    print(f"documents_loaded={len(documents)}")
    print(f"chunks_created={len(chunks)}")
    print(f"chunks_indexed={indexed_count}")
    print(f"results_returned={len(results)}")

    for index, result in enumerate(results, start=1):
        print(
            "result_{index}=file:{file} chunk_id:{chunk_id} preview:{preview}".format(
                index=index,
                file=result.metadata.get("file"),
                chunk_id=result.metadata.get("chunk_id"),
                preview=result.page_content[:120].replace("\n", " "),
            )
        )


if __name__ == "__main__":
    main()
