from typing import Any


class RagEngineError(Exception):
    """Base exception for domain-specific errors mapped to structured HTTP responses."""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or {}
        super().__init__(message)


class InvalidInputError(RagEngineError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="INVALID_INPUT", message=message, http_status=400, details=details
        )


class CorruptedPDFError(RagEngineError):
    def __init__(self, detail: str = "File is not a valid or readable PDF.") -> None:
        super().__init__(code="CORRUPTED_PDF", message=detail, http_status=422)


class EmptyPDFTextError(RagEngineError):
    def __init__(
        self,
        detail: str = (
            "No extractable text found in PDF. The document may be scanned/image-only "
            "and requires OCR before ingestion."
        ),
    ) -> None:
        super().__init__(code="EMPTY_PDF_TEXT", message=detail, http_status=422)


class DocumentNotFoundError(RagEngineError):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            code="DOCUMENT_NOT_FOUND",
            message=f"Document '{document_id}' was not found.",
            http_status=404,
        )


class EmbeddingProviderError(RagEngineError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            code="EMBEDDING_PROVIDER_ERROR",
            message=f"Embedding generation failed: {detail}",
            http_status=503,
        )