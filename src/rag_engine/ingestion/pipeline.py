import hashlib
import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_engine.config import Settings
from rag_engine.db.models import Document, DocumentChunk
from rag_engine.exceptions import DocumentNotFoundError, InvalidInputError
from rag_engine.ingestion.chunker import chunk_extracted_document
from rag_engine.ingestion.extractor import extract_pdf_bytes
from rag_engine.retrieval.embedder import EmbeddingProvider
from rag_engine.schemas.ingestion import (
    DocumentSummary,
    IngestionDiagnosticsResponse,
    IngestionResponse,
)

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        db: Session,
        embedder: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings

    def ingest_pdf(
        self,
        *,
        filename: str,
        file_bytes: bytes,
        doc_type: str = "general",
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResponse:
        if not filename or not filename.lower().endswith(".pdf"):
            raise InvalidInputError("Only .pdf files are supported.")

        max_bytes = self._settings.max_upload_size_mb * 1024 * 1024
        if not file_bytes:
            raise InvalidInputError("Uploaded file is empty.")
        if len(file_bytes) > max_bytes:
            raise InvalidInputError(
                f"File exceeds maximum allowed size of {self._settings.max_upload_size_mb} MB."
            )

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 1. Idempotency / Deduplication check
        existing = self._db.execute(
            select(Document).where(Document.file_hash == file_hash)
        ).scalar_one_or_none()

        if existing is not None:
            if existing.status == "ready":
                chunk_count = (
                    self._db.execute(
                        select(func.count(DocumentChunk.id)).where(
                            DocumentChunk.document_id == existing.id
                        )
                    ).scalar()
                    or 0
                )
                logger.info(
                    "Deduplicated upload filename=%s document_id=%s",
                    filename,
                    existing.id,
                )
                return IngestionResponse(
                    document_id=existing.id,
                    filename=existing.filename,
                    file_hash=existing.file_hash,
                    doc_type=existing.doc_type,
                    source=existing.source,
                    total_pages=existing.total_pages,
                    chunks_created=chunk_count,
                    status=existing.status,
                    deduplicated=True,
                )
            # Clean up prior failed attempt so re-upload works cleanly
            self._db.delete(existing)
            self._db.flush()

        # 2. Extract & Chunk (raises CorruptedPDFError / EmptyPDFTextError on bad input)
        extracted = extract_pdf_bytes(
            file_bytes, min_extracted_chars=self._settings.min_extracted_chars
        )
        drafts = chunk_extracted_document(
            extracted,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        # 3. Generate embeddings in CPU-friendly batches
        vectors = self._embedder.embed_passages([d.content for d in drafts])
        if len(vectors) != len(drafts):
            raise RuntimeError("Embedding count mismatch with chunk count.")

        # 4. Atomic insert of Document + all DocumentChunks
        doc = Document(
            filename=filename,
            file_hash=file_hash,
            doc_type=doc_type.strip().lower() or "general",
            source=source.strip() if source else None,
            total_pages=extracted.total_pages,
            metadata_json=metadata or {},
            status="ready",
        )

        for draft, vec in zip(drafts, vectors, strict=True):
            doc.chunks.append(
                DocumentChunk(
                    chunk_index=draft.chunk_index,
                    page_number=draft.page_number,
                    page_end=draft.page_end,
                    section_title=draft.section_title,
                    content=draft.content,
                    token_count=draft.token_count,
                    char_count=draft.char_count,
                    chunk_type=draft.chunk_type,
                    metadata_json=draft.metadata_json,
                    embedding=vec,
                )
            )

        self._db.add(doc)
        self._db.commit()
        self._db.refresh(doc)

        logger.info(
            "Ingested document_id=%s filename=%s pages=%d chunks=%d",
            doc.id,
            doc.filename,
            doc.total_pages,
            len(drafts),
        )

        return IngestionResponse(
            document_id=doc.id,
            filename=doc.filename,
            file_hash=doc.file_hash,
            doc_type=doc.doc_type,
            source=doc.source,
            total_pages=doc.total_pages,
            chunks_created=len(drafts),
            status=doc.status,
            deduplicated=False,
            diagnostics=IngestionDiagnosticsResponse(
                tables_detected=extracted.diagnostics.tables_detected,
                stripped_margin_blocks=extracted.diagnostics.stripped_margin_blocks,
                empty_pages=extracted.diagnostics.empty_pages,
            ),
        )

    def list_documents(
        self, *, doc_type: str | None = None, limit: int = 20, offset: int = 0
    ) -> tuple[int, list[DocumentSummary]]:
        base_filter = select(Document)
        count_stmt = select(func.count(Document.id))

        if doc_type:
            normalized_type = doc_type.strip().lower()
            base_filter = base_filter.where(Document.doc_type == normalized_type)
            count_stmt = count_stmt.where(Document.doc_type == normalized_type)

        total = self._db.execute(count_stmt).scalar() or 0
        docs = (
            self._db.execute(
                base_filter.order_by(Document.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )

        summaries: list[DocumentSummary] = []
        for doc in docs:
            chunk_count = (
                self._db.execute(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.document_id == doc.id
                    )
                ).scalar()
                or 0
            )
            summaries.append(
                DocumentSummary(
                    id=doc.id,
                    filename=doc.filename,
                    file_hash=doc.file_hash,
                    doc_type=doc.doc_type,
                    source=doc.source,
                    total_pages=doc.total_pages,
                    status=doc.status,
                    metadata_json=doc.metadata_json,
                    created_at=doc.created_at,
                    chunk_count=chunk_count,
                )
            )
        return total, summaries

    def get_document(self, document_id: uuid.UUID) -> DocumentSummary:
        doc = self._db.get(Document, document_id)
        if doc is None:
            raise DocumentNotFoundError(str(document_id))

        chunk_count = (
            self._db.execute(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_id == doc.id
                )
            ).scalar()
            or 0
        )
        return DocumentSummary(
            id=doc.id,
            filename=doc.filename,
            file_hash=doc.file_hash,
            doc_type=doc.doc_type,
            source=doc.source,
            total_pages=doc.total_pages,
            status=doc.status,
            metadata_json=doc.metadata_json,
            created_at=doc.created_at,
            chunk_count=chunk_count,
        )

    def delete_document(self, document_id: uuid.UUID) -> None:
        doc = self._db.get(Document, document_id)
        if doc is None:
            raise DocumentNotFoundError(str(document_id))
        self._db.delete(doc)
        self._db.commit()