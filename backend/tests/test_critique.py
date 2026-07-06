import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.agents import critique_agent  # noqa: E402


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return FakeResponse(self.response_text)


def _base_state():
    return {
        "query": "What does RefineRAG do?",
        "refined_query": None,
        "documents": [],
        "context": "RefineRAG is a multi-agent RAG system.",
        "answer": "It answers questions using retrieved context.",
        "critique": None,
        "retrieval_score": None,
        "answer_score": None,
        "iteration": 0,
        "max_iterations": 2,
        "answer_retry_count": 0,
        "sources": [],
    }


def test_critique_retrieval_node_returns_score_critique_and_refined_query(monkeypatch):
    fake_llm = FakeLLM(
        '{"score": 0.31, "critique": "The context is only loosely related.", "refined_query": "refine rag critique agent"}'
    )
    monkeypatch.setattr(critique_agent, "_get_llm", lambda: fake_llm)

    result = critique_agent.critique_retrieval_node(_base_state())

    assert result["retrieval_score"] == 0.31
    assert result["critique"] == "The context is only loosely related."
    assert result["refined_query"] == "refine rag critique agent"
    assert "Evaluate whether the retrieved context is relevant" in fake_llm.last_prompt


def test_critique_answer_node_returns_score_and_critique(monkeypatch):
    fake_llm = FakeLLM(
        '{"score": 0.92, "critique": "The answer is grounded in the provided context."}'
    )
    monkeypatch.setattr(critique_agent, "_get_llm", lambda: fake_llm)

    result = critique_agent.critique_answer_node(_base_state())

    assert result["answer_score"] == 0.92
    assert result["critique"] == "The answer is grounded in the provided context."
    assert "Check whether the answer is supported by the context." in fake_llm.last_prompt


def test_critique_helpers_clamp_and_fallback():
    assert critique_agent._clamp_score(1.7) == 1.0
    assert critique_agent._clamp_score(-0.2) == 0.0
    assert critique_agent._clamp_score("bad") == 0.0
    assert critique_agent._extract_json_payload("no json here") == {
        "score": 0.0,
        "critique": "Critique response could not be parsed.",
        "refined_query": "",
    }
