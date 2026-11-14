import logging
import math
from sqlalchemy.orm import Session

from rag_engine.config import Settings
from rag_engine.retrieval.embedder import EmbeddingProvider
from rag_engine.retrieval.query_utils import prepare_queries
from rag_engine.retrieval.reranker import RerankerProvider
from rag_engine.retrieval.search import HybridSearcher, SearchCandidate
from rag_engine.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedEvidence,
    ScoreBreakdown,
)

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MSG = (
    "The available documents do not contain sufficient relevant information "
    "to answer this query."
)


def _sigmoid(logit: float) -> float:
    clamped = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-clamped))


def build_constructed_context(
    evidences: list[RetrievedEvidence], max_chars: int
) -> str:
    """Assemble a citation-tagged context string bounded by max_chars."""
    blocks: list[str] = []
    total_len = 0

    for item in evidences:
        page_label = (
            f"{item.page_number}"
            if not item.page_end or item.page_end == item.page_number
            else f"{item.page_number}-{item.page_end}"
        )
        section_label = f" | Section: {item.section_title}" if item.section_title else ""
        header = (
            f"{item.citation_id} Document: {item.document_name} "
            f"| Page: {page_label}{section_label}"
        )
        block = f"{header}\n{item.content}"

        added_len = len(block) + (2 if blocks else 0)
        if blocks and (total_len + added_len) > max_chars:
            break
        if not blocks and len(block) > max_chars:
            blocks.append(block[:max_chars])
            break

        blocks.append(block)
        total_len += added_len

    return "\n\n".join(blocks)


class RetrievalService:
    def __init__(
        self,
        db: Session,
        embedder: EmbeddingProvider,
        reranker: RerankerProvider | None,
        settings: Settings,
    ) -> None:
        self._searcher = HybridSearcher(db)
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

    def retrieve(self, req: RetrievalRequest) -> RetrievalResponse:
        query = req.query.strip()
        keyword_query, rerank_query = prepare_queries(query)
        use_reranker = (
            req.use_reranker
            if req.use_reranker is not None
            else self._settings.enable_reranker
        ) and (self._reranker is not None)

        query_vec = self._embedder.embed_query(rerank_query)

        candidates = self._searcher.search(
            query_text=query,
            keyword_query=keyword_query,
            query_vector=query_vec,
            candidate_k=max(self._settings.retrieval_candidate_k, req.top_k),
            doc_type=req.doc_type,
            source=req.source,
            document_ids=req.document_ids,
            metadata_filters=req.metadata_filters,
        )

        if not candidates:
            logger.info("Retrieval returned 0 candidates for query.")
            return RetrievalResponse(
                query=query,
                sufficient_evidence=False,
                message=INSUFFICIENT_EVIDENCE_MSG,
                reranker_used=use_reranker,
                total_candidates_evaluated=0,
                results=[],
                constructed_context="",
            )

        # Tuple: (candidate, final_sort_score, gate_score, rerank_score)
        scored: list[tuple[SearchCandidate, float, float, float | None]] = []

        if use_reranker and self._reranker is not None:
            pool_size = max(self._settings.rerank_top_n, req.top_k)
            pool = candidates[:pool_size]
            max_rrf = max((c.rrf_score for c in pool), default=1.0) or 1.0
            rerank_scores = self._reranker.score_pairs(
                rerank_query, [c.content for c in pool]
            )
            for cand, r_score in zip(pool, rerank_scores, strict=True):
                rrf_norm = cand.rrf_score / max_rrf
                blended = 0.70 * _sigmoid(r_score) + 0.30 * rrf_norm
                scored.append((cand, blended, r_score, r_score))
            scored.sort(key=lambda item: item[1], reverse=True)
            threshold = self._settings.min_rerank_score
        else:
            for cand in candidates:
                base_score = (
                    cand.vector_score if cand.vector_score is not None else cand.rrf_score
                )
                scored.append((cand, base_score, base_score, None))
            threshold = self._settings.min_vector_score

        # Enforce strict sufficiency gate on gate_score (rerank logit or vector sim)
        # Enforce strict sufficiency gate:
        # - If reranker is on, accept if rerank_score >= threshold OR if a table chunk
        #   has strong exact keyword + semantic evidence (avoids prose-reranker false vetoes).
        if use_reranker:
            passing = [
                item
                for item in scored
                if item[2] >= threshold
                or (
                    item[0].chunk_type == "table"
                    and (item[0].keyword_score or 0.0) >= 0.25
                    and (item[0].vector_score or 0.0) >= 0.50
                )
            ]
        else:
            passing = [item for item in scored if item[2] >= threshold]

        if not passing:
            top_seen = max(item[2] for item in scored)
            logger.info(
                "Retrieval refused due to low relevance: top_score=%.4f threshold=%.4f",
                top_seen,
                threshold,
            )
            return RetrievalResponse(
                query=query,
                sufficient_evidence=False,
                message=INSUFFICIENT_EVIDENCE_MSG,
                reranker_used=use_reranker,
                total_candidates_evaluated=len(candidates),
                results=[],
                constructed_context="",
            )

        selected = passing[: req.top_k]
        evidences: list[RetrievedEvidence] = []
        for idx, (cand, final_score, _gate, r_score) in enumerate(selected, start=1):
            evidences.append(
                RetrievedEvidence(
                    citation_id=f"[{idx}]",
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    document_name=cand.document_name,
                    doc_type=cand.doc_type,
                    page_number=cand.page_number,
                    page_end=cand.page_end,
                    section_title=cand.section_title,
                    chunk_type=cand.chunk_type,
                    content=cand.content,
                    relevance_score=round(final_score, 4),
                    scores=ScoreBreakdown(
                        vector_score=(
                            round(cand.vector_score, 4)
                            if cand.vector_score is not None
                            else None
                        ),
                        keyword_score=(
                            round(cand.keyword_score, 4)
                            if cand.keyword_score is not None
                            else None
                        ),
                        rrf_score=round(cand.rrf_score, 6),
                        rerank_score=round(r_score, 4) if r_score is not None else None,
                    ),
                )
            )

        context_str = build_constructed_context(
            evidences, max_chars=self._settings.max_context_chars
        )

        return RetrievalResponse(
            query=query,
            sufficient_evidence=True,
            message=f"Retrieved {len(evidences)} grounded evidence chunk(s).",
            reranker_used=use_reranker,
            total_candidates_evaluated=len(candidates),
            results=evidences,
            constructed_context=context_str,
        )