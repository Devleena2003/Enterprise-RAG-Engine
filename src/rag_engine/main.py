import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from rag_engine.api.health import router as health_router
from rag_engine.api.ingest import router as ingest_router
from rag_engine.config import get_settings
from rag_engine.exceptions import RagEngineError
from rag_engine.logging import configure_logging, request_id_ctx
from rag_engine.schemas.common import ErrorDetail, ErrorResponse
from rag_engine.api.retrieve import router as retrieve_router

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
            logger.exception(
                "Unhandled exception processing %s %s", request.method, request.url.path
            )
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

    @app.exception_handler(RagEngineError)
    async def rag_engine_exception_handler(
        request: Request, exc: RagEngineError
    ) -> JSONResponse:
        req_id = request_id_ctx.get()
        payload = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                request_id=req_id,
                details=exc.details,
            )
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=payload.model_dump(),
            headers={"X-Request-ID": req_id},
        )

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(retrieve_router)
    return app


app = create_app()