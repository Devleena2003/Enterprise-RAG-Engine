from typing import Any
import uuid
from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = Field(default=None)
    source: str | None = Field(default=None)
    document_ids: list[uuid.UUID] | None = Field(default=None)
    metadata_filters: dict[str, Any] | None = Field(default=None)
    use_reranker: bool | None = Field(
        default=None,
        description="Override server default reranker setting for A/B comparison.",
    )


class ScoreBreakdown(BaseModel):
    vector_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float
    rerank_score: float | None = None


class RetrievedEvidence(BaseModel):
    """A single grounded chunk with mandatory non-fabricable source attribution."""

    citation_id: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    doc_type: str
    page_number: int
    page_end: int | None = None
    section_title: str | None = None
    chunk_type: str
    content: str
    relevance_score: float
    scores: ScoreBreakdown


class RetrievalResponse(BaseModel):
    query: str
    sufficient_evidence: bool
    message: str
    reranker_used: bool
    total_candidates_evaluated: int
    results: list[RetrievedEvidence]
    constructed_context: str