from typing import List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    critique: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    chunks_created: int


class ResetIndexResponse(BaseModel):
    message: str
