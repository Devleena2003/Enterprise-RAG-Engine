import fitz
from fastapi.testclient import TestClient

from rag_engine.main import create_app
from rag_engine.retrieval.embedder import get_embedding_provider
from rag_engine.retrieval.reranker import get_reranker_provider


class KeywordAwareFakeEmbedder:
    """Deterministic 384-dim embedder so cosine similarity reflects topic overlap in tests."""

    @property
    def dimension(self) -> int:
        return 384

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        v = [0.01] * 384
        if "flood" in low or "deductible" in low or "14b" in low:
            v[0] = 0.95
        if "tax" in low or "1099" in low:
            v[1] = 0.95
        return v

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


class KeywordAwareFakeReranker:
    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        q_words = set(query.lower().split())
        scores: list[float] = []
        for doc in documents:
            overlap = sum(1 for w in q_words if len(w) > 2 and w in doc.lower())
            scores.append(2.5 if overlap > 0 else -6.0)
        return scores


def _make_pdf(title: str, body: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 90), title, fontsize=13)
    page.insert_text((50, 130), body, fontsize=11)
    b = doc.tobytes()
    doc.close()
    return b


def test_hybrid_retrieve_with_filters_citations_and_refusal() -> None:
    app = create_app()
    app.dependency_overrides[get_embedding_provider] = lambda: KeywordAwareFakeEmbedder()
    app.dependency_overrides[get_reranker_provider] = lambda: KeywordAwareFakeReranker()
    client = TestClient(app)

    p1 = _make_pdf(
        "Section 4.2 Deductibles",
        "Flood damage requires endorsement Clause 14B with a $1,000 deductible.",
    )
    p2 = _make_pdf(
        "Section 1.1 Tax Reporting",
        "Form 1099-NEC must be filed by January 31 for contractor payments.",
    )

    r1 = client.post(
        "/api/v1/documents",
        files={"file": ("flood_policy_test.pdf", p1, "application/pdf")},
        data={"doc_type": "insurance_policy", "metadata_json": '{"region": "east"}'},
    )
    r2 = client.post(
        "/api/v1/documents",
        files={"file": ("tax_guide_test.pdf", p2, "application/pdf")},
        data={"doc_type": "financial", "metadata_json": '{"region": "west"}'},
    )
    doc_id_1 = r1.json()["document_id"]
    doc_id_2 = r2.json()["document_id"]

    try:
        # 1. Exact keyword + semantic retrieval with metadata filter
        res = client.post(
            "/api/v1/retrieve",
            json={
                "query": "What does Clause 14B say about flood deductible?",
                "doc_type": "insurance_policy",
                "metadata_filters": {"region": "east"},
                "top_k": 3,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["sufficient_evidence"] is True
        assert len(body["results"]) >= 1
        top = body["results"][0]
        assert top["citation_id"] == "[1]"
        assert top["document_name"] == "flood_policy_test.pdf"
        assert top["page_number"] == 1
        assert "Clause 14B" in top["content"]
        assert "[1] Document: flood_policy_test.pdf | Page: 1" in body["constructed_context"]

        # 2. Off-topic query must refuse rather than fabricate citations
        off_topic = client.post(
            "/api/v1/retrieve",
            json={
                "query": "Explain quantum computing qubits",
                "document_ids": [doc_id_1, doc_id_2],
            },
        )
        assert off_topic.status_code == 200
        assert off_topic.json()["sufficient_evidence"] is False
        assert off_topic.json()["results"] == []
    finally:
        client.delete(f"/api/v1/documents/{doc_id_1}")
        client.delete(f"/api/v1/documents/{doc_id_2}")