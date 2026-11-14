from datetime import datetime
import re

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it",
    "many", "much", "of", "on", "or", "that", "the", "their", "there",
    "these", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "will", "with", "happened", "tell", "give", "show", "find",
}

_DATE_PATTERNS = [
    # e.g., "1st August 2023", "01 Aug 2023", "1 august, 2023"
    re.compile(
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4})\b",
        re.IGNORECASE,
    ),
    # e.g., "August 1st, 2023", "Aug 1 2023"
    re.compile(
        r"\b([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})\b",
        re.IGNORECASE,
    ),
    # e.g., "01-08-2023", "01/08/2023", "01-08-23", "2023-08-01"
    re.compile(r"\b(\d{1,4}[-/]\d{1,2}[-/]\d{2,4})\b"),
]

_PARSE_FORMATS = [
    "%d %B %Y", "%d %b %Y", "%d %B %y", "%d %b %y",
    "%B %d %Y", "%b %d %Y", "%B %d %y", "%b %d %y",
    "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
    "%Y-%m-%d", "%Y/%m/%d",
]


def _parse_date_candidate(raw: str) -> datetime | None:
    cleaned = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " ")
    cleaned = " ".join(cleaned.split())
    for fmt in _PARSE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def extract_date_variants(query: str) -> list[str]:
    """Detect dates in natural text and return common enterprise document formats."""
    variants: list[str] = []
    seen: set[str] = set()

    for pattern in _DATE_PATTERNS:
        for match in pattern.findall(query):
            dt = _parse_date_candidate(match)
            if dt is None:
                continue
            candidates = [
                dt.strftime("%d-%m-%y"),
                dt.strftime("%d-%m-%Y"),
                dt.strftime("%d/%m/%y"),
                dt.strftime("%d/%m/%Y"),
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%d %b %Y"),
            ]
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    variants.append(c)
    return variants


def prepare_queries(raw_query: str) -> tuple[str, str]:
    """
    Return (websearch_or_query, rerank_query):
    - websearch_or_query: safe OR-joined terms + quoted date variants for Postgres tsvector
    - rerank_query: original query enriched with normalized date hints if present
    """
    date_variants = extract_date_variants(raw_query)

    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", raw_query)
    terms: list[str] = []
    seen_lower: set[str] = set()

    for dv in date_variants:
        low = dv.lower()
        if low not in seen_lower:
            seen_lower.add(low)
            terms.append(f'"{dv}"')

    for tok in raw_tokens:
        low = tok.lower()
        if low in _STOPWORDS or len(low) < 2 or low in seen_lower:
            continue
        seen_lower.add(low)
        if "-" in tok or "/" in tok:
            terms.append(f'"{tok}"')
        else:
            terms.append(tok)

    websearch_or_query = " OR ".join(terms) if terms else raw_query

    if date_variants:
        hint = ", ".join(date_variants[:3])
        rerank_query = f"{raw_query} ({hint})"
    else:
        rerank_query = raw_query

    return websearch_or_query, rerank_query