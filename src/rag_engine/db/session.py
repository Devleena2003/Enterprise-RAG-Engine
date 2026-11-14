import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from rag_engine.config import get_settings

logger = logging.getLogger(__name__)


def create_db_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_readiness() -> tuple[bool, str]:
    """Verify PostgreSQL connectivity and that the pgvector extension is installed."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            ext = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            if ext != 1:
                return False, "pgvector extension is not installed in target database"
        return True, "ok"
    except SQLAlchemyError as exc:
        logger.warning("Database readiness check failed: %s", exc)
        return False, "database_unreachable"