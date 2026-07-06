from typing import Any, List, Optional, TypedDict


class RAGState(TypedDict):
    query: str
    refined_query: Optional[str]
    documents: List[Any]
    context: str
    answer: Optional[str]
    critique: Optional[str]
    retrieval_score: Optional[float]
    answer_score: Optional[float]
    iteration: int
    max_iterations: int
    answer_retry_count: int
    sources: List[dict[str, Any]]
