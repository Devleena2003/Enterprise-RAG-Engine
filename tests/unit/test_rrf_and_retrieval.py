import uuid
from rag_engine.config import Settings
from rag_engine.retrieval.search import SearchCandidate, reciprocal_rank_fusion
from rag_engine.retrieval.service import INSUFFICIENT_EVIDENCE_MSG, RetrievalService
from rag_engine.schemas.retrieval import RetrievalRequest


def _cand(cid: uuid.UUID, text: str, v: float | None = None, k: float | None = None):
    return SearchCandidate(
        chunk_id=cid,
        document_id=uuid.uuid4(),
        document_name="policy.pdf",
        doc_type="insurance_policy",
        page_number=3,
        page_end=3,
        section_title="Section 3.1",
        chunk_type="text",
        content=text,
        vector_score=v,
        keyword_score=k,
    )


def test_rrf_boosts_chunk_present_in_both_lists() -> None:
    c_both = uuid.uuid4()
    c_vec_only = uuid.uuid4()
    c_kw_only = uuid.uuid4()

    vec_list = [_cand(c_vec_only, "v", v=0.8), _cand(c_both, "both", v=0.75)]
    kw_list = [_cand(c_kw_only, "k", k=0.9), _cand(c_both, "both", k=0.85)]

    fused = reciprocal_rank_fusion(vec_list, kw_list, k=60)
    assert fused[0].chunk_id == c_both
    assert fused[0].vector_score == 0.75
    assert fused[0].keyword_score == 0.85


class _StubEmbedder:
    @property
    def dimension(self) -> int:
        return 384

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 384


class _LowScoreReranker:
    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        return [-7.5 for _ in documents]


class _CloseScoreReranker:
    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        return [1.6, 1.5]


def test_retrieval_refuses_when_rerank_score_below_threshold(monkeypatch) -> None:
    c1 = _cand(uuid.uuid4(), "Unrelated bank header text.", v=0.45)
    service = RetrievalService(
        db=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        reranker=_LowScoreReranker(),
        settings=Settings(enable_reranker=True, min_rerank_score=-2.0),
    )
    monkeypatch.setattr(service._searcher, "search", lambda **kwargs: [c1])

    resp = service.retrieve(RetrievalRequest(query="Who won the World Cup?"))

    assert resp.reranker_used is True
    assert resp.sufficient_evidence is False
    assert resp.message == INSUFFICIENT_EVIDENCE_MSG
    assert resp.results == []
    assert resp.constructed_context == ""


def test_retrieval_without_reranker_refuses_low_vector_score(monkeypatch) -> None:
    c1 = _cand(uuid.uuid4(), "Unrelated bank header text.", v=0.45)
    service = RetrievalService(
        db=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        reranker=_LowScoreReranker(),
        settings=Settings(enable_reranker=False, min_vector_score=0.55),
    )
    monkeypatch.setattr(service._searcher, "search", lambda **kwargs: [c1])

    resp = service.retrieve(RetrievalRequest(query="Who won the World Cup?"))

    assert resp.reranker_used is False
    assert resp.sufficient_evidence is False
    assert resp.results == []


def test_blended_scoring_preserves_strong_hybrid_rrf_match(monkeypatch) -> None:
    c_vec_only = _cand(uuid.uuid4(), "04-08-23 balance", v=0.79, k=None)
    c_vec_only.rrf_score = 0.015

    c_hybrid_hit = _cand(uuid.uuid4(), "01-08-23 balance", v=0.78, k=2.4)
    c_hybrid_hit.rrf_score = 0.032

    service = RetrievalService(
        db=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        reranker=_CloseScoreReranker(),
        settings=Settings(enable_reranker=True, min_rerank_score=-2.5),
    )
    monkeypatch.setattr(
        service._searcher, "search", lambda **kwargs: [c_vec_only, c_hybrid_hit]
    )

    resp = service.retrieve(RetrievalRequest(query="01-08-23 balance"))

    assert resp.sufficient_evidence is True
    assert resp.results[0].chunk_id == c_hybrid_hit.chunk_id
    assert resp.results[0].citation_id == "[1]"


def test_table_with_strong_keyword_match_not_vetoed_by_reranker(monkeypatch) -> None:
    table_cand = SearchCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="sbi.pdf",
        doc_type="statement",
        page_number=1,
        page_end=1,
        section_title="REGULAR SAVINGS BANK ACCOUNT",
        chunk_type="table",
        content="| 01-08-23 | NEFT | 164211.00 CR | 171191.04 CR |",
        vector_score=0.78,
        keyword_score=2.4,
        rrf_score=0.032,
    )

    service = RetrievalService(
        db=None,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        reranker=_LowScoreReranker(),
        settings=Settings(enable_reranker=True, min_rerank_score=-2.5),
    )
    monkeypatch.setattr(service._searcher, "search", lambda **kwargs: [table_cand])

    resp = service.retrieve(
        RetrievalRequest(query="1st august 2023 transactions", use_reranker=True)
    )

    assert resp.sufficient_evidence is True
    assert len(resp.results) == 1
    assert resp.results[0].page_number == 1