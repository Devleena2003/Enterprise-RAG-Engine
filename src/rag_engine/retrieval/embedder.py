import logging
from functools import lru_cache
from typing import Protocol

from rag_engine.config import get_settings
from rag_engine.exceptions import EmbeddingProviderError

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Interface for embedding models so providers remain swappable."""

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class FastEmbedProvider:
    """
    Lightweight local ONNX embedding provider using fastembed.
    Lazy-loads the model on first inference call so API startup stays instant.
    """

    def __init__(
        self,
        model_name: str,
        dimension: int,
        batch_size: int = 16,
        threads: int = 2,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._threads = threads
        self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                logger.info(
                    "Loading local ONNX embedding model: %s (threads=%d)",
                    self._model_name,
                    self._threads,
                )
                self._model = TextEmbedding(
                    model_name=self._model_name,
                    threads=self._threads,
                )
            except Exception as exc:
                raise EmbeddingProviderError(str(exc)) from exc
        return self._model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            vectors = list(model.passage_embed(texts, batch_size=self._batch_size))
            return [vec.tolist() for vec in vectors]
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(str(exc)) from exc

    def embed_query(self, query: str) -> list[float]:
        try:
            model = self._get_model()
            vec = next(model.query_embed([query]))
            return vec.tolist()
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(str(exc)) from exc


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    return FastEmbedProvider(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        threads=settings.embedding_threads,
    )