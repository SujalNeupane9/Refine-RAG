import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.graph import workflow  # noqa: E402


def test_workflow_completes_with_retrieval_and_answer_retry(monkeypatch):
    call_counts = {"retrieve": 0, "critique_retrieval": 0, "generate": 0, "critique_answer": 0}

    def fake_retrieve(state):
        call_counts["retrieve"] += 1
        return {
            **state,
            "iteration": state.get("iteration", 0) + 1,
            "documents": ["chunk"],
            "context": "retrieved context",
            "sources": [{"file": "sample.md", "page": 1, "chunk_id": "0"}],
        }

    def fake_critique_retrieval(state):
        call_counts["critique_retrieval"] += 1
        score = 0.2 if state["iteration"] < 2 else 0.9
        refined_query = "refined query" if score < 0.7 else None
        return {
            **state,
            "retrieval_score": score,
            "critique": "retrieval critique",
            "refined_query": refined_query,
        }

    def fake_generate(state):
        call_counts["generate"] += 1
        answer_retry_count = state.get("answer_retry_count", 0)
        if state.get("answer"):
            answer_retry_count += 1
        answer = "draft answer" if call_counts["generate"] == 1 else "final answer"
        return {
            **state,
            "answer_retry_count": answer_retry_count,
            "answer": answer,
        }

    def fake_critique_answer(state):
        call_counts["critique_answer"] += 1
        score = 0.4 if state.get("answer_retry_count", 0) == 0 else 0.9
        return {
            **state,
            "answer_score": score,
            "critique": "answer critique",
        }

    monkeypatch.setattr(workflow, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(workflow, "critique_retrieval_node", fake_critique_retrieval)
    monkeypatch.setattr(workflow, "generate_node", fake_generate)
    monkeypatch.setattr(workflow, "critique_answer_node", fake_critique_answer)

    graph = workflow.build_workflow()
    final_state = graph.invoke(
        {
            "query": "What is RefineRAG?",
            "refined_query": None,
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
    )

    assert final_state["answer"] == "final answer"
    assert final_state["iteration"] == 2
    assert final_state["answer_retry_count"] == 1
    assert final_state["answer_score"] == 0.9
    assert call_counts == {
        "retrieve": 2,
        "critique_retrieval": 2,
        "generate": 2,
        "critique_answer": 2,
    }


def test_route_retrieval_and_answer_decisions():
    assert workflow.route_retrieval({"retrieval_score": 0.8, "iteration": 1, "max_iterations": 2}) == "generate"
    assert workflow.route_retrieval({"retrieval_score": 0.2, "iteration": 1, "max_iterations": 2}) == "retrieve"
    assert workflow.route_retrieval({"retrieval_score": 0.2, "iteration": 2, "max_iterations": 2}) == "generate"

    assert workflow.route_answer({"answer_score": 0.8, "answer_retry_count": 0}) == "end"
    assert workflow.route_answer({"answer_score": 0.2, "answer_retry_count": 0}) == "generate"
    assert workflow.route_answer({"answer_score": 0.2, "answer_retry_count": 1}) == "end"
