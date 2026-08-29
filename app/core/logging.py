import logging
import os
from typing import Any

import structlog
from structlog import BoundLogger

# Service configuration
SERVICE_NAME = os.getenv("SERVICE_NAME", "ai-lesson-builder-service")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup structlog with JSON output following StandardLogTemplate."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", level=getattr(logging, log_level.upper(), logging.INFO))


def get_logger(name: str | None = None) -> BoundLogger:
    """Return structlog logger with service/environment context."""
    logger = structlog.get_logger(name)
    # Bind default context fields
    logger = logger.bind(
        service=SERVICE_NAME,
        environment=ENVIRONMENT,
    )
    return logger


# Feature mapping based on endpoint path
FEATURE_MAP = {
    "/api/v1/lessons": "LESSON",
    "/api/v1/health": "SYSTEM",
    "/api/v1/profile": "USER",
    "/api/v1/memory": "MEMORY",
    "/ui": "SYSTEM",
}


def get_feature_from_path(path: str) -> str:
    """Map endpoint path to feature name."""
    for prefix, feature in FEATURE_MAP.items():
        if path.startswith(prefix):
            return feature
    return "UNKNOWN"


def _to_string(value: Any) -> str:
    """Convert value to string for JSON logging."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)
