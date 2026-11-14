import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Async-safe storage for the current request's correlation ID
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for easy parsing and aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_ctx.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_payload, ensure_ascii=False)


def configure_logging(log_level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Replace existing handlers to avoid duplicate logs under uvicorn
    root_logger.handlers.clear()
    root_logger.addHandler(handler)