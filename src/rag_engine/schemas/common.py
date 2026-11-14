from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: str
    service: str


class ReadinessResponse(BaseModel):
    status: str
    service: str
    checks: dict[str, str]