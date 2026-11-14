class RagEngineError(Exception):
    """Base exception for domain-specific errors."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class CorruptedPDFError(RagEngineError):
    """Raised when uploaded bytes cannot be opened as a valid PDF."""

    def __init__(self, detail: str = "File is not a valid or readable PDF.") -> None:
        super().__init__(code="CORRUPTED_PDF", message=detail)


class EmptyPDFTextError(RagEngineError):
    """Raised when a PDF contains too little extractable text (e.g., scanned images)."""

    def __init__(
        self,
        detail: str = (
            "No extractable text found in PDF. The document may be scanned/image-only "
            "and requires OCR before ingestion."
        ),
    ) -> None:
        super().__init__(code="EMPTY_PDF_TEXT", message=detail)