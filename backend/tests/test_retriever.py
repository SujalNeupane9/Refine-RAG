import sys
from pathlib import Path

from langchain_core.documents import Document


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.agents import retriever_agent  # noqa: E402


def test_retrieve_node_returns_documents_context_and_sources(monkeypatch):
    captured = {}
    documents = [
        Document(
            page_content="The Critique Agent checks context relevance.",
            metadata={"file": "sample.md", "page": 1, "chunk_id": "0"},
        ),
        Document(
            page_content="The Generator Agent answers from approved context.",
            metadata={"file": "sample.md", "page": 1, "chunk_id": "1"},
        ),
    ]

    def fake_search(query: str, top_k: int):
        captured["query"] = query
        captured["top_k"] = top_k
        return documents

    monkeypatch.setattr(retriever_agent, "search", fake_search)

    state = {
        "query": "What does the critique agent check?",
        "refined_query": "critique agent context relevance",
        "documents": [],
        "context": "",
        "answer": None,
        "critique": None,
        "retrieval_score": None,
        "answer_score": None,
        "iteration": 0,
        "max_iterations": 2,
        "answer_retry_count": 0,
        "sources": [],
    }

    result = retriever_agent.retrieve_node(state)

    assert captured["query"] == "critique agent context relevance"
    assert captured["top_k"] == retriever_agent.TOP_K
    assert result["documents"] == documents
    assert "The Critique Agent checks context relevance." in result["context"]
    assert "[Source 1: sample.md, page 1, chunk 0]" in result["context"]
    assert result["sources"] == [
        {
            "file": "sample.md",
            "page": 1,
            "chunk_id": "0",
            "chunk_text": "The Critique Agent checks context relevance.",
        },
        {
            "file": "sample.md",
            "page": 1,
            "chunk_id": "1",
            "chunk_text": "The Generator Agent answers from approved context.",
        },
    ]


def test_format_docs_and_extract_sources_deduplicate_sources():
    documents = [
        Document(
            page_content="First chunk.",
            metadata={"file": "guide.md", "page": 2, "chunk_id": "a"},
        ),
        Document(
            page_content="Duplicate source chunk.",
            metadata={"file": "guide.md", "page": 2, "chunk_id": "a"},
        ),
    ]

    context = retriever_agent.format_docs(documents)
    sources = retriever_agent.extract_sources(documents)

    assert "[Source 1: guide.md, page 2, chunk a]" in context
    assert "Duplicate source chunk." in context
    assert sources == [
        {"file": "guide.md", "page": 2, "chunk_id": "a", "chunk_text": "First chunk."}
    ]
