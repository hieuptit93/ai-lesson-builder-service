from datetime import datetime, timezone

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AIServiceError,
    ExtractionError,
    LLMProviderError,
    LowQualityImageError,
    RateLimitExceededError,
    UnsafeContentError,
)
from app.core.logging import get_feature_from_path

logger = structlog.get_logger()

STATUS_MAP: dict[type, int] = {
    UnsafeContentError: 422,
    ExtractionError: 422,
    LowQualityImageError: 422,
    LLMProviderError: 502,
    RateLimitExceededError: 429,
}

# Client errors (4xx) should use WARN, server errors (5xx) use ERROR
CLIENT_ERRORS = (UnsafeContentError, ExtractionError, LowQualityImageError)


async def ai_service_exception_handler(request: Request, exc: AIServiceError) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID", "unknown")
    correlation_id = request.headers.get("X-Correlation-ID", request_id)
    status_code = STATUS_MAP.get(type(exc), 500)
    feature = get_feature_from_path(str(request.url.path))

    # Determine log level: WARN for client errors (4xx), ERROR for server errors (5xx)
    is_client_error = isinstance(exc, CLIENT_ERRORS)
    event = "api.request.client_error" if is_client_error else "api.request.server_error"

    log_func = logger.warning if is_client_error else logger.error
    log_func(
        event,
        log_type="api",
        feature=feature,
        http_method=request.method,
        endpoint=str(request.url.path),
        status_code=status_code,
        error_type=type(exc).__name__,
        error_message=exc.message,
        request_id=request_id,
        correlation_id=correlation_id,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "status": "error",
            "error": {
                "code": exc.error_code,
                "message": exc.message,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID", "unknown")
    correlation_id = request.headers.get("X-Correlation-ID", request_id)
    feature = get_feature_from_path(str(request.url.path))

    logger.error(
        "api.request.server_error",
        log_type="api",
        feature=feature,
        http_method=request.method,
        endpoint=str(request.url.path),
        status_code=500,
        error_type=type(exc).__name__,
        error_message=str(exc),
        request_id=request_id,
        correlation_id=correlation_id,
        exc=str(exc),  # This will be rendered as stack_trace by structlog
    )

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "status": "error",
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
