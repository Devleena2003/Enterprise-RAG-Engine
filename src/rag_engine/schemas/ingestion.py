import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngestionDiagnosticsResponse(BaseModel):
    tables_detected: int
    stripped_margin_blocks: int
    empty_pages: list[int] = Field(default_factory=list)


class IngestionResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    file_hash: str
    doc_type: str
    source: str | None
    total_pages: int
    chunks_created: int
    status: str
    deduplicated: bool
    diagnostics: IngestionDiagnosticsResponse | None = None


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_hash: str
    doc_type: str
    source: str | None
    total_pages: int
    status: str
    metadata_json: dict[str, Any]
    created_at: datetime
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentSummary]


class DeleteDocumentResponse(BaseModel):
    document_id: uuid.UUID
    deleted: bool