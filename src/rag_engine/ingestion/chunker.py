import re
from dataclasses import dataclass, field
from typing import Any

from rag_engine.ingestion.extractor import ExtractedDocument

# Heuristic patterns for enterprise headings ("Section 4.2 Deductible", "1.2 Coverage", "ARTICLE IV")
_HEADING_PATTERNS = [
    re.compile(r"^(?:SECTION|ARTICLE|CHAPTER|CLAUSE|PART|APPENDIX)\s+[A-Z0-9IVX.]+.*$", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{2,90}$"),
]


@dataclass
class ChunkDraft:
    """In-memory chunk ready for embedding and database persistence."""

    chunk_index: int
    page_number: int
    page_end: int
    section_title: str | None
    content: str
    chunk_type: str  # "text" | "table"
    token_count: int
    char_count: int
    metadata_json: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Lightweight CPU-free token approximation (~4 chars/token for English)."""
    return max(1, len(text) // 4)


def detect_heading(block_text: str) -> str | None:
    """Detect whether a short text block looks like a section heading."""
    candidate = block_text.strip()
    if not candidate or len(candidate) > 110 or "\n" in candidate:
        return None
    if candidate.endswith((".", ",", ";")):
        return None

    for pattern in _HEADING_PATTERNS:
        if pattern.match(candidate):
            return candidate

    # Short ALL-CAPS heading like "EXCLUSIONS AND LIMITATIONS"
    alpha_chars = [c for c in candidate if c.isalpha()]
    if len(alpha_chars) >= 4 and candidate.isupper():
        return candidate

    return None


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries (fallback to hard char split)."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                parts.append(sentence[i : i + max_chars].strip())
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = sentence

    if current:
        parts.append(current)
    return parts


def _split_large_markdown_table(table_md: str, max_chars: int) -> list[str]:
    """Keep tables intact when possible; if oversized, split by rows repeating header."""
    if len(table_md) <= max_chars:
        return [table_md]

    lines = table_md.splitlines()
    if len(lines) <= 2:
        return [table_md]

    header_block = f"{lines[0]}\n{lines[1]}"
    data_rows = lines[2:]
    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = len(header_block)

    for row in data_rows:
        added_len = len(row) + 1
        if current_rows and (current_len + added_len) > max_chars:
            chunks.append(header_block + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_len = len(header_block) + added_len
        else:
            current_rows.append(row)
            current_len += added_len

    if current_rows:
        chunks.append(header_block + "\n" + "\n".join(current_rows))
    return chunks


def _tail_overlap(text: str, overlap_chars: int) -> str:
    """Take up to overlap_chars from the end of text without cutting mid-word."""
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text
    snippet = text[-overlap_chars:]
    first_space = snippet.find(" ")
    if first_space != -1 and first_space < len(snippet) - 1:
        return snippet[first_space + 1 :]
    return snippet


def chunk_extracted_document(
    extracted: ExtractedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[ChunkDraft]:
    """
    Convert extracted blocks into structure-aware ChunkDrafts preserving:
    - Exact page_number / page_end attribution
    - Active section_title
    - Separate intact Markdown table chunks
    - Overlap within the same section
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

    drafts: list[ChunkDraft] = []
    current_section: str | None = None

    # Accumulator for prose text chunks
    acc_parts: list[str] = []
    acc_start_page: int | None = None
    acc_end_page: int | None = None

    def flush_prose(carry_overlap: bool = True) -> None:
        nonlocal acc_parts, acc_start_page, acc_end_page
        if not acc_parts or acc_start_page is None or acc_end_page is None:
            acc_parts = []
            acc_start_page = None
            acc_end_page = None
            return

        content = "\n\n".join(acc_parts).strip()
        if content:
            drafts.append(
                ChunkDraft(
                    chunk_index=len(drafts),
                    page_number=acc_start_page,
                    page_end=acc_end_page,
                    section_title=current_section,
                    content=content,
                    chunk_type="text",
                    token_count=estimate_tokens(content),
                    char_count=len(content),
                )
            )
            overlap = _tail_overlap(content, chunk_overlap) if carry_overlap else ""
            if overlap:
                acc_parts = [overlap]
                acc_start_page = acc_end_page
            else:
                acc_parts = []
                acc_start_page = None
                acc_end_page = None
        else:
            acc_parts = []
            acc_start_page = None
            acc_end_page = None

    for block in extracted.blocks:
        if block.block_type == "table":
            # Flush pending prose before emitting table so types never mix
            flush_prose(carry_overlap=False)
            for table_part in _split_large_markdown_table(block.content, chunk_size):
                drafts.append(
                    ChunkDraft(
                        chunk_index=len(drafts),
                        page_number=block.page_number,
                        page_end=block.page_number,
                        section_title=current_section,
                        content=table_part,
                        chunk_type="table",
                        token_count=estimate_tokens(table_part),
                        char_count=len(table_part),
                    )
                )
            continue

        # Check if this text block is a section heading
        heading = detect_heading(block.content)
        if heading:
            flush_prose(carry_overlap=False)
            current_section = heading
            acc_parts = [heading]
            acc_start_page = block.page_number
            acc_end_page = block.page_number
            continue

        # Normal prose block: split if a single paragraph exceeds chunk_size
        for sub_para in _split_long_paragraph(block.content, chunk_size):
            current_text = "\n\n".join(acc_parts)
            projected_len = (
                len(current_text) + 2 + len(sub_para) if current_text else len(sub_para)
            )

            if acc_parts and projected_len > chunk_size:
                flush_prose(carry_overlap=True)
                # Edge case: if overlap + sub_para still exceeds chunk_size, drop overlap
                if acc_parts and (len(acc_parts[0]) + 2 + len(sub_para)) > chunk_size:
                    acc_parts = []
                    acc_start_page = None
                    acc_end_page = None

            if acc_start_page is None:
                acc_start_page = block.page_number
            acc_end_page = block.page_number
            acc_parts.append(sub_para)

    flush_prose(carry_overlap=False)
    return drafts