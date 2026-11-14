from dataclasses import dataclass, field

import fitz  # PyMuPDF

from rag_engine.exceptions import CorruptedPDFError, EmptyPDFTextError
from rag_engine.ingestion.cleaner import (
    clean_extracted_text,
    detect_repeating_margin_lines,
    is_noise_margin_text,
)


@dataclass
class ExtractedBlock:
    """A single ordered content unit (text block or Markdown table) on a PDF page."""

    page_number: int
    block_type: str  # "text" | "table"
    content: str


@dataclass
class ExtractionDiagnostics:
    """Audit metrics explaining what the extractor kept, cleaned, or skipped."""

    tables_detected: int = 0
    stripped_margin_blocks: int = 0
    repeating_margin_signatures: list[str] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    total_pages: int
    blocks: list[ExtractedBlock] = field(default_factory=list)
    diagnostics: ExtractionDiagnostics = field(default_factory=ExtractionDiagnostics)


def _format_table_as_markdown(rows: list[list[str | None]]) -> str:
    """Convert extracted table cells into a clean Markdown table."""
    cleaned_rows: list[list[str]] = []
    for row in rows:
        cleaned_cells = [
            " ".join((cell or "").replace("|", "\\|").split()) for cell in row
        ]
        if any(cleaned_cells):
            cleaned_rows.append(cleaned_cells)

    if not cleaned_rows:
        return ""

    col_count = max(len(r) for r in cleaned_rows)
    padded_rows = [r + [""] * (col_count - len(r)) for r in cleaned_rows]

    header = "| " + " | ".join(padded_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(r) + " |" for r in padded_rows[1:]]
    return "\n".join([header, separator, *body])


def _block_overlaps_any_table(
    block_rect: fitz.Rect, table_rects: list[fitz.Rect]
) -> bool:
    """Check whether >50% of a text block's area sits inside a detected table."""
    block_area = block_rect.get_area()
    if block_area <= 0:
        return False
    for t_rect in table_rects:
        intersection = block_rect & t_rect
        if not intersection.is_empty and (intersection.get_area() / block_area) > 0.5:
            return True
    return False


def extract_pdf_bytes(
    pdf_bytes: bytes, min_extracted_chars: int = 30
) -> ExtractedDocument:
    """
    Extract structured text blocks and Markdown tables from raw PDF bytes,
    filtering out repeating headers/footers and page numbers.
    """
    if not pdf_bytes:
        raise CorruptedPDFError("Uploaded PDF file is empty (0 bytes).")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptedPDFError(f"Failed to parse PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise CorruptedPDFError("PDF contains zero pages.")

        page_raw_items: list[list[tuple[float, str, str, bool]]] = []
        margin_texts_by_page: list[list[str]] = []
        tables_detected = 0

        for page in doc:
            page_height = page.rect.height or 792.0
            top_cutoff = page_height * 0.12
            bottom_cutoff = page_height * 0.88

            table_rects: list[fitz.Rect] = []
            items: list[tuple[float, str, str, bool]] = []
            page_margins: list[str] = []

            table_finder = page.find_tables()
            for table in table_finder.tables:
                t_rect = fitz.Rect(table.bbox)
                md = _format_table_as_markdown(table.extract())
                if md:
                    table_rects.append(t_rect)
                    items.append((t_rect.y0, "table", md, False))
                    tables_detected += 1

            for block in page.get_text("blocks"):
                x0, y0, x1, y1, raw_text, _block_no, b_type = block
                if b_type != 0 or not raw_text or not raw_text.strip():
                    continue
                b_rect = fitz.Rect(x0, y0, x1, y1)
                if _block_overlaps_any_table(b_rect, table_rects):
                    continue

                is_margin = (y1 <= top_cutoff) or (y0 >= bottom_cutoff)
                if is_margin:
                    page_margins.append(raw_text.strip())
                items.append((y0, "text", raw_text, is_margin))

            items.sort(key=lambda item: item[0])
            page_raw_items.append(items)
            margin_texts_by_page.append(page_margins)

        repeating_sigs = detect_repeating_margin_lines(margin_texts_by_page)
        extracted_blocks: list[ExtractedBlock] = []
        total_chars = 0
        stripped_margin_blocks = 0
        empty_pages: list[int] = []

        for page_idx, items in enumerate(page_raw_items, start=1):
            page_chars = 0
            for _y0, item_type, raw_content, is_margin in items:
                if item_type == "text":
                    if is_margin and is_noise_margin_text(raw_content, repeating_sigs):
                        stripped_margin_blocks += 1
                        continue
                    cleaned = clean_extracted_text(raw_content)
                    if cleaned:
                        extracted_blocks.append(
                            ExtractedBlock(
                                page_number=page_idx,
                                block_type="text",
                                content=cleaned,
                            )
                        )
                        total_chars += len(cleaned)
                        page_chars += len(cleaned)
                else:
                    extracted_blocks.append(
                        ExtractedBlock(
                            page_number=page_idx,
                            block_type="table",
                            content=raw_content,
                        )
                    )
                    total_chars += len(raw_content)
                    page_chars += len(raw_content)

            if page_chars == 0:
                empty_pages.append(page_idx)

        if total_chars < min_extracted_chars:
            raise EmptyPDFTextError()

        diagnostics = ExtractionDiagnostics(
            tables_detected=tables_detected,
            stripped_margin_blocks=stripped_margin_blocks,
            repeating_margin_signatures=sorted(repeating_sigs),
            empty_pages=empty_pages,
        )
        return ExtractedDocument(
            total_pages=doc.page_count,
            blocks=extracted_blocks,
            diagnostics=diagnostics,
        )
    finally:
        doc.close()