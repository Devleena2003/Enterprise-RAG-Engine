import argparse
import sys
from pathlib import Path

from rag_engine.config import get_settings
from rag_engine.exceptions import RagEngineError
from rag_engine.ingestion.chunker import ChunkDraft, chunk_extracted_document
from rag_engine.ingestion.extractor import ExtractedDocument, extract_pdf_bytes


def render_review_markdown(
    pdf_path: Path, extracted: ExtractedDocument, chunks: list[ChunkDraft]
) -> str:
    """Build a human-auditable Markdown report for side-by-side PDF verification."""
    diag = extracted.diagnostics
    lines: list[str] = [
        f"# Extraction & Chunking Audit: `{pdf_path.name}`",
        "",
        "## Diagnostics",
        f"- **Total Pages:** {extracted.total_pages}",
        f"- **Extracted Blocks:** {len(extracted.blocks)}",
        f"- **Generated Chunks:** {len(chunks)}",
        f"- **Tables Detected:** {diag.tables_detected}",
        f"- **Stripped Header/Footer Blocks:** {diag.stripped_margin_blocks}",
        f"- **Repeating Margin Patterns:** {diag.repeating_margin_signatures or ['None']}",
        f"- **Pages With Zero Text (Possible Scans/Images):** {diag.empty_pages or ['None']}",
        "",
        "---",
        "",
        "## Generated Chunks",
        "",
    ]

    for chunk in chunks:
        page_label = (
            f"{chunk.page_number}"
            if chunk.page_number == chunk.page_end
            else f"{chunk.page_number}-{chunk.page_end}"
        )
        lines.extend(
    [
        (
            f"### Chunk #{chunk.chunk_index} | Page {page_label} | "
            f"Type: `{chunk.chunk_type}` | Section: `{chunk.section_title or 'None'}` | "
            f"Chars: {chunk.char_count} (~{chunk.token_count} tokens)"
        ),
        "",
        chunk.content,
    ]
)
    return "\n".join(lines)


def inspect_single_pdf(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    output_dir: Path | None = None,
) -> Path | None:
    """Run extraction + chunking on one PDF, print summary, and optionally save review Markdown."""
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return None

    print("=" * 70)
    print(f"Inspecting: {pdf_path}")

    try:
        extracted = extract_pdf_bytes(pdf_path.read_bytes())
        chunks = chunk_extracted_document(
            extracted, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    except RagEngineError as exc:
        print(f"[FAILED - {exc.code}] {exc.message}")
        return None

    diag = extracted.diagnostics
    print(
        f"Pages: {extracted.total_pages} | Chunks: {len(chunks)} | "
        f"Tables: {diag.tables_detected} | Stripped Margins: {diag.stripped_margin_blocks}"
    )
    if diag.repeating_margin_signatures:
        print(f"Removed Margin Signatures: {diag.repeating_margin_signatures}")
    if diag.empty_pages:
        print(f"[WARNING] Pages with 0 extracted text: {diag.empty_pages}")

    # Show first 3 chunks in terminal for quick spot check
    for chunk in chunks[:3]:
        preview = chunk.content.replace("\n", " ")[:180]
        print(
            f"  -> Chunk #{chunk.chunk_index} (p.{chunk.page_number}, "
            f"{chunk.chunk_type}, sec={chunk.section_title!r}): {preview}..."
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        review_path = output_dir / f"{pdf_path.stem}_review.md"
        review_path.write_text(
            render_review_markdown(pdf_path, extracted, chunks), encoding="utf-8"
        )
        print(f"Full audit written to: {review_path}")
        return review_path

    return None


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Dry-run PDF extraction and chunking inspector for sample_data PDFs."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to a PDF file. If omitted, inspects all *.pdf files inside sample_data/.",
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print terminal summary only without saving *_review.md.",
    )
    args = parser.parse_args(argv)

    output_dir = None if args.no_save else Path("sample_data/output")

    if args.pdf_path:
        targets = [Path(args.pdf_path)]
    else:
        sample_dir = Path("sample_data")
        targets = sorted(sample_dir.glob("*.pdf"))
        if not targets:
            print(
                "No PDF files found in sample_data/. "
                "Drop any PDF (e.g., sample_data/my_test.pdf) and re-run."
            )
            return 0

    for target in targets:
        inspect_single_pdf(
            target,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            output_dir=output_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())