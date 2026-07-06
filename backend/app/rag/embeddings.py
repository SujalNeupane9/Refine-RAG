import logging
from hashlib import sha256
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import EMBEDDING_MODEL, GOOGLE_API_KEY


logger = logging.getLogger(__name__)


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


class HybridEmbeddings(Embeddings):
    def __init__(self, primary: GoogleGenerativeAIEmbeddings, fallback: Embeddings):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._use_fallback:
            return self.fallback.embed_documents(texts)

        try:
            return self.primary.embed_documents(texts)
        except Exception:
            logger.exception("Gemini embeddings failed; falling back to local hash embeddings.")
            self._use_fallback = True
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._use_fallback:
            return self.fallback.embed_query(text)

        try:
            return self.primary.embed_query(text)
        except Exception:
            logger.exception("Gemini query embedding failed; falling back to local hash embeddings.")
            self._use_fallback = True
            return self.fallback.embed_query(text)


def get_embeddings() -> Embeddings:
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is missing; using local hash embeddings.")
        return LocalHashEmbeddings()

    logger.info("Initializing Gemini embeddings: model=%s", EMBEDDING_MODEL)
    return HybridEmbeddings(
        primary=GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        ),
        fallback=LocalHashEmbeddings(),
    )
