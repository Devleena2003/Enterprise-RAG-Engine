from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-rag-engine")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    database_url: str = Field(
        default="postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"
    )
    db_pool_size: int = Field(default=5, ge=1, le=20)
    db_max_overflow: int = Field(default=5, ge=0, le=20)

    # Embedding & Ingestion
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dimension: int = Field(default=384, ge=64, le=4096)
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_threads: int = Field(default=2, ge=1, le=8)

    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    min_extracted_chars: int = Field(default=30, ge=1)
    max_upload_size_mb: int = Field(default=20, ge=1, le=100)

    # Hybrid Retrieval & Reranking
    # Hybrid Retrieval & Reranking
    reranker_model_name: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")
    enable_reranker: bool = Field(default=False)
    reranker_threads: int = Field(default=2, ge=1, le=8)
    retrieval_candidate_k: int = Field(default=20, ge=5, le=100)
    rerank_top_n: int = Field(default=12, ge=1, le=50)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    min_rerank_score: float = Field(default=-2.5)
    min_vector_score: float = Field(default=0.55, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=500, le=32000)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()