import json
import uuid

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from rag_engine.config import get_settings
from rag_engine.db.session import get_db
from rag_engine.exceptions import InvalidInputError
from rag_engine.ingestion.pipeline import IngestionService
from rag_engine.retrieval.embedder import EmbeddingProvider, get_embedding_provider
from rag_engine.schemas.ingestion import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummary,
    IngestionResponse,
)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


def get_ingestion_service(
    db: Session = Depends(get_db),
    embedder: EmbeddingProvider = Depends(get_embedding_provider),
) -> IngestionService:
    return IngestionService(db=db, embedder=embedder, settings=get_settings())


@router.post("", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    doc_type: str = Form(default="general"),
    source: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    parsed_metadata = {}
    if metadata_json and metadata_json.strip():
        try:
            loaded = json.loads(metadata_json)
            if not isinstance(loaded, dict):
                raise InvalidInputError("metadata_json must be a JSON object.")
            parsed_metadata = loaded
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"Invalid metadata_json: {exc}") from exc

    file_bytes = await file.read()
    result = service.ingest_pdf(
        filename=file.filename or "upload.pdf",
        file_bytes=file_bytes,
        doc_type=doc_type,
        source=source,
        metadata=parsed_metadata,
    )
    if result.deduplicated:
        response.status_code = status.HTTP_200_OK
    return result


@router.get("", response_model=DocumentListResponse)
def list_documents(
    doc_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    total, items = service.list_documents(doc_type=doc_type, limit=limit, offset=offset)
    return DocumentListResponse(total=total, items=items)


@router.get("/{document_id}", response_model=DocumentSummary)
def get_document(
    document_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentSummary:
    return service.get_document(document_id)


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteDocumentResponse:
    service.delete_document(document_id)
    return DeleteDocumentResponse(document_id=document_id, deleted=True)