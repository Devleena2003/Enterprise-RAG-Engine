from dataclasses import dataclass
import json
from typing import Any
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class SearchCandidate:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    doc_type: str
    page_number: int
    page_end: int | None
    section_title: str | None
    chunk_type: str
    content: str
    vector_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float = 0.0


def reciprocal_rank_fusion(
    vector_ranked: list[SearchCandidate],
    keyword_ranked: list[SearchCandidate],
    k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[SearchCandidate]:
    """
    Fuse two ranked candidate lists using Reciprocal Rank Fusion:
    RRF(d) = sum( weight / (k + rank) ), where rank is 1-indexed.
    """
    merged: dict[uuid.UUID, SearchCandidate] = {}

    for rank, cand in enumerate(vector_ranked, start=1):
        entry = merged.get(cand.chunk_id)
        if entry is None:
            entry = cand
            merged[cand.chunk_id] = entry
        else:
            entry.vector_score = cand.vector_score
        entry.rrf_score += vector_weight / (k + rank)

    for rank, cand in enumerate(keyword_ranked, start=1):
        entry = merged.get(cand.chunk_id)
        if entry is None:
            entry = cand
            merged[cand.chunk_id] = entry
        else:
            entry.keyword_score = cand.keyword_score
        entry.rrf_score += keyword_weight / (k + rank)

    fused = list(merged.values())
    fused.sort(key=lambda item: item.rrf_score, reverse=True)
    return fused


class HybridSearcher:
    """Executes filtered pgvector semantic search and tsvector keyword search."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def _build_filter_clause(
        self,
        *,
        doc_type: str | None,
        source: str | None,
        document_ids: list[uuid.UUID] | None,
        metadata_filters: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        clauses = ["d.status = 'ready'"]
        params: dict[str, Any] = {}

        if doc_type:
            clauses.append("d.doc_type = :doc_type")
            params["doc_type"] = doc_type.strip().lower()

        if source:
            clauses.append("d.source = :source")
            params["source"] = source.strip()

        if document_ids:
            clauses.append("d.id = ANY(CAST(:doc_ids AS uuid[]))")
            params["doc_ids"] = [str(uid) for uid in document_ids]

        if metadata_filters:
            clauses.append("d.metadata_json @> CAST(:meta_filters AS jsonb)")
            params["meta_filters"] = json.dumps(metadata_filters)

        return " AND ".join(clauses), params

    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        keyword_query: str | None = None,
        candidate_k: int = 20,
        doc_type: str | None = None,
        source: str | None = None,
        document_ids: list[uuid.UUID] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchCandidate]:
        where_sql, filter_params = self._build_filter_clause(
            doc_type=doc_type,
            source=source,
            document_ids=document_ids,
            metadata_filters=metadata_filters,
        )

        vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"

        # 1. Semantic vector search (cosine similarity = 1 - cosine distance)
        vector_sql = text(
            f"""
            SELECT
                c.id AS chunk_id,
                d.id AS document_id,
                d.filename AS document_name,
                d.doc_type AS doc_type,
                c.page_number,
                c.page_end,
                c.section_title,
                c.chunk_type,
                c.content,
                1.0 - (c.embedding <=> CAST(:q_vec AS vector)) AS vector_score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY c.embedding <=> CAST(:q_vec AS vector) ASC
            LIMIT :limit
            """
        )
        vec_rows = self._db.execute(
            vector_sql,
            {**filter_params, "q_vec": vector_literal, "limit": candidate_k},
        ).mappings()

        vector_candidates = [
            SearchCandidate(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                doc_type=row["doc_type"],
                page_number=row["page_number"],
                page_end=row["page_end"],
                section_title=row["section_title"],
                chunk_type=row["chunk_type"],
                content=row["content"],
                vector_score=float(row["vector_score"]),
            )
            for row in vec_rows
        ]

        # 2. Full-text keyword search using websearch_to_tsquery
        keyword_sql = text(
            f"""
            SELECT
                c.id AS chunk_id,
                d.id AS document_id,
                d.filename AS document_name,
                d.doc_type AS doc_type,
                c.page_number,
                c.page_end,
                c.section_title,
                c.chunk_type,
                c.content,
                ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :q_text)) AS keyword_score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
              AND c.search_vector @@ websearch_to_tsquery('english', :q_text)
            ORDER BY keyword_score DESC
            LIMIT :limit
            """
        )
        kw_query = keyword_query or query_text
        kw_rows = self._db.execute(
            keyword_sql,
            {**filter_params, "q_text": kw_query, "limit": candidate_k},
        ).mappings()

        keyword_candidates = [
            SearchCandidate(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                doc_type=row["doc_type"],
                page_number=row["page_number"],
                page_end=row["page_end"],
                section_title=row["section_title"],
                chunk_type=row["chunk_type"],
                content=row["content"],
                keyword_score=float(row["keyword_score"]),
            )
            for row in kw_rows
        ]

        return reciprocal_rank_fusion(vector_candidates, keyword_candidates)