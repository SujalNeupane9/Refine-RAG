import logging

from langgraph.graph import END, START, StateGraph

from app.agents.critique_agent import critique_answer_node, critique_retrieval_node
from app.agents.generator_agent import generate_node
from app.agents.retriever_agent import retrieve_node
from app.graph.state import RAGState


logger = logging.getLogger(__name__)


def route_retrieval(state: RAGState) -> str:
    score = state.get("retrieval_score") or 0.0
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 0)

    if score >= 0.7:
        logger.info(
            "Routing after retrieval critique: score=%s iteration=%s decision=generate",
            score,
            iteration,
        )
        return "generate"

    if iteration < max_iterations:
        logger.info(
            "Routing after retrieval critique: score=%s iteration=%s max_iterations=%s decision=retrieve",
            score,
            iteration,
            max_iterations,
        )
        return "retrieve"

    logger.info(
        "Routing after retrieval critique: score=%s iteration=%s max_iterations=%s decision=generate",
        score,
        iteration,
        max_iterations,
    )
    return "generate"


def route_answer(state: RAGState) -> str:
    score = state.get("answer_score") or 0.0
    retry_count = state.get("answer_retry_count", 0)

    if score >= 0.7:
        logger.info(
            "Routing after answer critique: score=%s retry_count=%s decision=end",
            score,
            retry_count,
        )
        return "end"

    if retry_count < 1:
        logger.info(
            "Routing after answer critique: score=%s retry_count=%s decision=generate",
            score,
            retry_count,
        )
        return "generate"

    logger.info(
        "Routing after answer critique: score=%s retry_count=%s decision=end",
        score,
        retry_count,
    )
    return "end"


def build_workflow():
    logger.info("Building RefineRAG LangGraph workflow.")

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("critique_retrieval", critique_retrieval_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critique_answer", critique_answer_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "critique_retrieval")
    workflow.add_conditional_edges(
        "critique_retrieval",
        route_retrieval,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", "critique_answer")
    workflow.add_conditional_edges(
        "critique_answer",
        route_answer,
        {
            "generate": "generate",
            "end": END,
        },
    )

    compiled = workflow.compile()
    logger.info("RefineRAG LangGraph workflow compiled successfully.")
    return compiled


rag_graph = build_workflow()
