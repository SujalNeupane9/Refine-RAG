import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.agents import generator_agent  # noqa: E402


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return FakeResponse(self.content)


def test_generate_node_returns_answer_and_uses_grounded_prompt(monkeypatch):
    fake_llm = FakeLLM("The Critique Agent checks relevance and grounding.")
    monkeypatch.setattr(generator_agent, "_get_llm", lambda: fake_llm)

    state = {
        "query": "What does the Critique Agent do?",
        "refined_query": None,
        "documents": [],
        "context": "The Critique Agent checks whether retrieved context is useful and whether generated answers are grounded.",
        "answer": None,
        "critique": None,
        "retrieval_score": None,
        "answer_score": None,
        "iteration": 0,
        "max_iterations": 2,
        "answer_retry_count": 0,
        "sources": [{"file": "sample.md", "page": 1, "chunk_id": "0"}],
    }

    result = generator_agent.generate_node(state)

    assert result["answer"] == "The Critique Agent checks relevance and grounding."
    assert "Do not make up facts." in fake_llm.last_prompt
    assert "The Critique Agent checks whether retrieved context is useful" in fake_llm.last_prompt


def test_build_generation_prompt_mentions_insufficient_context():
    prompt = generator_agent.build_generation_prompt(
        query="What is the document about?",
        context="",
    )

    assert "If the context is insufficient, say so clearly." in prompt
    assert "using only the context below" in prompt
