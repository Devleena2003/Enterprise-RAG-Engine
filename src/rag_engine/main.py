import logging
import time
import uuid

from fastapi import FastAPI, Request, Response

from rag_engine.api.health import router as health_router
from rag_engine.config import get_settings
from rag_engine.logging import configure_logging, request_id_ctx

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Enterprise Ingestion & Retrieval Engine",
        version="0.1.0",
        description="Modular document ingestion and hybrid retrieval service with strict citations.",
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        token = request_id_ctx.set(req_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled exception processing %s %s", request.method, request.url.path)
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "HTTP request completed: method=%s path=%s duration_ms=%s",
                request.method,
                request.url.path,
                duration_ms,
            )
            request_id_ctx.reset(token)

        response.headers["X-Request-ID"] = req_id
        return response

    app.include_router(health_router)
    return app


app = create_app()