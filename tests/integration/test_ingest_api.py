import fitz
from fastapi.testclient import TestClient

from rag_engine.main import create_app
from rag_engine.retrieval.embedder import get_embedding_provider


class FakeEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 384

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.02] * 384 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.02] * 384


def _make_sample_pdf(unique_tag: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Section 1.1 Overview", fontsize=13)
    page.insert_text(
        (50, 140),
        f"Enterprise compliance test document {unique_tag}. Coverage limit is $100,000.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_upload_deduplicate_list_and_delete_document() -> None:
    app = create_app()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    client = TestClient(app)

    pdf_bytes = _make_sample_pdf("REF-9981")

    # 1. Upload valid PDF -> 201 Created
    res1 = client.post(
        "/api/v1/documents",
        files={"file": ("test_policy.pdf", pdf_bytes, "application/pdf")},
        data={
            "doc_type": "insurance_policy",
            "source": "underwriting",
            "metadata_json": '{"year": 2025}',
        },
    )
    assert res1.status_code == 201
    body1 = res1.json()
    assert body1["deduplicated"] is False
    assert body1["chunks_created"] >= 1
    assert body1["doc_type"] == "insurance_policy"
    doc_id = body1["document_id"]

    try:
        # 2. Upload identical PDF again -> 200 OK & deduplicated=True
        res2 = client.post(
            "/api/v1/documents",
            files={"file": ("test_policy_copy.pdf", pdf_bytes, "application/pdf")},
        )
        assert res2.status_code == 200
        assert res2.json()["deduplicated"] is True
        assert res2.json()["document_id"] == doc_id

        # 3. Verify GET list & GET by ID
        list_res = client.get("/api/v1/documents?doc_type=insurance_policy")
        assert list_res.status_code == 200
        assert any(item["id"] == doc_id for item in list_res.json()["items"])

        get_res = client.get(f"/api/v1/documents/{doc_id}")
        assert get_res.status_code == 200
        assert get_res.json()["chunk_count"] == body1["chunks_created"]

        # 4. Invalid metadata_json returns structured ErrorResponse
        bad_meta = client.post(
            "/api/v1/documents",
            files={"file": ("x.pdf", pdf_bytes, "application/pdf")},
            data={"metadata_json": "{not valid json"},
        )
        assert bad_meta.status_code == 400
        assert bad_meta.json()["error"]["code"] == "INVALID_INPUT"
    finally:
        # 5. Clean up test document
        del_res = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200

        after_del = client.get(f"/api/v1/documents/{doc_id}")
        assert after_del.status_code == 404
        assert after_del.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"