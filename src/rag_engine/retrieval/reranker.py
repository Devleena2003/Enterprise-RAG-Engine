from functools import lru_cache
import logging
from typing import Protocol

from rag_engine.config import get_settings
from rag_engine.exceptions import RagEngineError

logger = logging.getLogger(__name__)


class RerankerProviderError(RagEngineError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            code="RERANKER_PROVIDER_ERROR",
            message=f"Reranking failed: {detail}",
            http_status=503,
        )


class RerankerProvider(Protocol):
    """Interface for cross-encoder rerankers."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """
    Lightweight local ONNX cross-encoder reranker.
    Lazy-loaded so API startup stays instant and zero-RAM until needed.
    """

    def __init__(self, model_name: str, threads: int = 2) -> None:
        self._model_name = model_name
        self._threads = threads
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info(
                    "Loading local ONNX cross-encoder reranker: %s (threads=%d)",
                    self._model_name,
                    self._threads,
                )
                self._model = TextCrossEncoder(
                    model_name=self._model_name,
                    threads=self._threads,
                )
            except Exception as exc:
                raise RerankerProviderError(str(exc)) from exc
        return self._model

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        try:
            model = self._get_model()
            return [float(s) for s in model.rerank(query, documents)]
        except RerankerProviderError:
            raise
        except Exception as exc:
            raise RerankerProviderError(str(exc)) from exc


@lru_cache(maxsize=1)
def get_reranker_provider() -> RerankerProvider:
    settings = get_settings()
    return FastEmbedReranker(
        model_name=settings.reranker_model_name,
        threads=settings.reranker_threads,
    )