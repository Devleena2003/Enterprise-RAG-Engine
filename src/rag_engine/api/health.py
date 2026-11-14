from fastapi import APIRouter, Response, status

from rag_engine.config import get_settings
from rag_engine.db.session import check_db_readiness
from rag_engine.schemas.common import LivenessResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=LivenessResponse)
def health_live() -> LivenessResponse:
    settings = get_settings()
    return LivenessResponse(status="ok", service=settings.app_name)


@router.get("/ready", response_model=ReadinessResponse)
def health_ready(response: Response) -> ReadinessResponse:
    settings = get_settings()
    db_ok, db_detail = check_db_readiness()

    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="degraded",
            service=settings.app_name,
            checks={"database": db_detail},
        )

    return ReadinessResponse(
        status="ready",
        service=settings.app_name,
        checks={"database": db_detail},
    )