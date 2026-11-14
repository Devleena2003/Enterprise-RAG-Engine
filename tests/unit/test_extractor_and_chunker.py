from pathlib import Path

import fitz
import pytest

from rag_engine.exceptions import CorruptedPDFError, EmptyPDFTextError
from rag_engine.ingestion.chunker import chunk_extracted_document
from rag_engine.ingestion.cleaner import clean_extracted_text
from rag_engine.ingestion.extractor import extract_pdf_bytes
from rag_engine.ingestion.inspect_pdf import inspect_single_pdf


def _build_messy_policy_pdf() -> bytes:
    """Create an in-memory multi-page PDF with headers, footers, hyphens, and a grid table."""
    doc = fitz.open()

    # Page 1: Header, section heading, hyphenated wrap, footer
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 35), "ACME INSURANCE CORP - CONFIDENTIAL", fontsize=9)
    p1.insert_text((50, 110), "Section 1.1 Coverage Scope", fontsize=13)
    p1.insert_text(
        (50, 145),
        "This policy covers sudden accidental water dam-\nage inside the insured residence.",
        fontsize=11,
    )
    p1.insert_text((250, 810), "Page 1 of 2", fontsize=9)

    # Page 2: Same repeating header/footer + a real drawn grid table + Section 2.1
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 35), "ACME INSURANCE CORP - CONFIDENTIAL", fontsize=9)
    p2.insert_text((50, 110), "Section 2.1 Deductible Schedule", fontsize=13)

    x_cols = [50, 200, 380]
    y_rows = [140, 170, 200, 230]
    shape = p2.new_shape()
    for y in y_rows:
        shape.draw_line((x_cols[0], y), (x_cols[-1], y))
    for x in x_cols:
        shape.draw_line((x, y_rows[0]), (x, y_rows[-1]))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()

    table_cells = [
        ("Tier", "Deductible"),
        ("Standard", "$1,000"),
        ("Platinum", "$250"),
    ]
    for r_idx, (c1, c2) in enumerate(table_cells):
        y_text = y_rows[r_idx] + 20
        p2.insert_text((60, y_text), c1, fontsize=10)
        p2.insert_text((210, y_text), c2, fontsize=10)

    p2.insert_text(
        (50, 280),
        "Claims exceeding the deductible must be reported within 30 days under Clause 14B.",
        fontsize=11,
    )
    p2.insert_text((250, 810), "Page 2 of 2", fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_clean_repairs_hyphenation_and_whitespace() -> None:
    raw = "Accidental water dam-\nage\ninside the\u00a0building.\n\n\nSecond   paragraph."
    cleaned = clean_extracted_text(raw)
    assert cleaned == "Accidental water damage inside the building.\n\nSecond paragraph."


def test_extract_strips_repeating_headers_footers_and_records_diagnostics() -> None:
    pdf_bytes = _build_messy_policy_pdf()
    extracted = extract_pdf_bytes(pdf_bytes)

    assert extracted.total_pages == 2
    assert extracted.diagnostics.tables_detected == 1
    assert extracted.diagnostics.stripped_margin_blocks >= 4
    assert extracted.diagnostics.empty_pages == []

    full_text = "\n".join(b.content for b in extracted.blocks)
    assert "ACME INSURANCE CORP - CONFIDENTIAL" not in full_text
    assert "Page 1 of 2" not in full_text
    assert "Page 2 of 2" not in full_text
    assert "damage" in full_text

    table_blocks = [b for b in extracted.blocks if b.block_type == "table"]
    assert len(table_blocks) == 1
    assert "| Tier | Deductible |" in table_blocks[0].content
    assert "| Platinum | $250 |" in table_blocks[0].content
    assert table_blocks[0].page_number == 2


def test_chunk_preserves_page_numbers_sections_and_tables() -> None:
    pdf_bytes = _build_messy_policy_pdf()
    extracted = extract_pdf_bytes(pdf_bytes)
    chunks = chunk_extracted_document(extracted, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 3
    assert chunks[0].page_number == 1
    assert chunks[0].section_title == "Section 1.1 Coverage Scope"

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].page_number == 2
    assert table_chunks[0].section_title == "Section 2.1 Deductible Schedule"

    prose_after_table = [c for c in chunks if "Clause 14B" in c.content]
    assert len(prose_after_table) == 1
    assert prose_after_table[0].page_number == 2


def test_corrupted_and_blank_pdfs_raise_explicit_errors() -> None:
    with pytest.raises(CorruptedPDFError):
        extract_pdf_bytes(b"not a pdf at all")

    blank_doc = fitz.open()
    blank_doc.new_page()
    blank_bytes = blank_doc.tobytes()
    blank_doc.close()

    with pytest.raises(EmptyPDFTextError):
        extract_pdf_bytes(blank_bytes)


def test_inspect_single_pdf_writes_review_markdown(tmp_path: Path) -> None:
    pdf_file = tmp_path / "my_test.pdf"
    pdf_file.write_bytes(_build_messy_policy_pdf())
    out_dir = tmp_path / "output"

    review_path = inspect_single_pdf(
        pdf_file, chunk_size=500, chunk_overlap=50, output_dir=out_dir
    )

    assert review_path is not None
    assert review_path.exists()
    md_text = review_path.read_text(encoding="utf-8")
    assert "# Extraction & Chunking Audit: `my_test.pdf`" in md_text
    assert "**Tables Detected:** 1" in md_text
    assert "| Platinum | $250 |" in md_text