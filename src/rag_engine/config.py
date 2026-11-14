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
    embedding_dimension: int = Field(default=384, ge=64, le=4096)
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    min_extracted_chars: int = Field(default=30, ge=1)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()