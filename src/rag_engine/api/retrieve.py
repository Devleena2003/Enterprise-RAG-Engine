from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from rag_engine.config import get_settings
from rag_engine.db.session import get_db
from rag_engine.retrieval.embedder import EmbeddingProvider, get_embedding_provider
from rag_engine.retrieval.reranker import RerankerProvider, get_reranker_provider
from rag_engine.retrieval.service import RetrievalService
from rag_engine.schemas.retrieval import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


def get_retrieval_service(
    db: Session = Depends(get_db),
    embedder: EmbeddingProvider = Depends(get_embedding_provider),
    reranker: RerankerProvider = Depends(get_reranker_provider),
) -> RetrievalService:
    return RetrievalService(
        db=db,
        embedder=embedder,
        reranker=reranker,
        settings=get_settings(),
    )


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_evidence(
    payload: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    return service.retrieve(payload)