import re
import unicodedata
from collections import Counter

# Matches common standalone footer/header page indicators like "Page 3 of 20" or "- 12 -"
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|[-–—]?\s*\d{1,4}\s*[-–—]?)$",
    re.IGNORECASE,
)


def normalize_margin_signature(text: str) -> str:
    """Normalize digits so 'Page 1 of 10' and 'Page 2 of 10' share the same signature."""
    collapsed = " ".join(text.strip().lower().split())
    return re.sub(r"\d+", "#", collapsed)


def detect_repeating_margin_lines(
    margin_blocks_by_page: list[list[str]],
) -> set[str]:
    """
    Identify margin text signatures (headers/footers) that repeat across pages.
    Only activates for multi-page documents (>= 2 pages).
    """
    num_pages = len(margin_blocks_by_page)
    if num_pages < 2:
        return set()

    doc_freq: Counter[str] = Counter()
    for page_margins in margin_blocks_by_page:
        unique_sigs = {
            normalize_margin_signature(txt) for txt in page_margins if txt.strip()
        }
        for sig in unique_sigs:
            doc_freq[sig] += 1

    threshold = max(2, int(num_pages * 0.5))
    return {sig for sig, count in doc_freq.items() if count >= threshold}


def is_noise_margin_text(text: str, repeating_signatures: set[str]) -> bool:
    """Return True if a margin block is a page number or a repeating header/footer."""
    stripped = " ".join(text.strip().split())
    if not stripped:
        return True
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    return normalize_margin_signature(stripped) in repeating_signatures


def clean_extracted_text(text: str) -> str:
    """
    Normalize messy PDF text while preserving paragraph breaks:
    - Unicode NFKC normalization (fixes weird ligatures like 'ﬁ' -> 'fi')
    - Repairs soft hyphenation across line breaks ('compen-\\nsation' -> 'compensation')
    - Converts single line breaks inside paragraphs into spaces
    - Collapses excessive spaces and blank lines
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")

    # Repair hyphenated words split across PDF lines (e.g., "policy-\nholder")
    normalized = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", normalized)

    # Preserve intentional paragraph breaks (\n\n+), join single visual line wraps
    paragraphs = re.split(r"\n{2,}", normalized)
    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        joined = re.sub(r"\s*\n\s*", " ", para)
        joined = re.sub(r"[ \t]{2,}", " ", joined).strip()
        if joined:
            cleaned_paragraphs.append(joined)

    return "\n\n".join(cleaned_paragraphs)