import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_feature_from_path

logger = structlog.get_logger()


def _get_status_event(status_code: int) -> tuple[str, str]:
    """Map status code to log level and event name."""
    if 200 <= status_code < 300:
        return "INFO", "api.request.success"
    if 300 <= status_code < 400:
        return "INFO", "api.request.success"  # Redirects
    if 400 <= status_code < 500:
        return "WARN", "api.request.client_error"
    return "ERROR", "api.request.server_error"


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        request_id = request.headers.get("X-Request-ID", "")
        correlation_id = request.headers.get("X-Correlation-ID", request_id)

        # Bind correlation context
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
        )

        # Log request start
        feature = get_feature_from_path(str(request.url.path))
        logger.info(
            "api.request.start",
            log_type="api",
            feature=feature,
            http_method=request.method,
            endpoint=str(request.url.path),
            request_id=request_id,
            correlation_id=correlation_id,
        )

        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Log response
        level, event = _get_status_event(response.status_code)

        log_func = logger.info if level == "INFO" else logger.warning if level == "WARN" else logger.error
        log_func(
            event,
            log_type="api",
            feature=feature,
            http_method=request.method,
            endpoint=str(request.url.path),
            status_code=response.status_code,
            duration_ms=elapsed_ms,
            request_id=request_id,
            correlation_id=correlation_id,
        )

        # Metric-event RED (stream log_type:metric) -> log-based metric
        try:
            from app.core.metric_events import (
                emit as metric_emit,
                outcome_for as metric_outcome_for,
                route_template as metric_route_template,
                spine_from_headers as metric_spine_from_headers,
            )

            metric_emit(
                "http_request",
                endpoint=metric_route_template(request),
                method=request.method,
                status_code=response.status_code,
                outcome=metric_outcome_for(response.status_code),
                duration_ms=float((time.monotonic() - start) * 1000.0),
                **metric_spine_from_headers(request.headers),
            )
        except Exception:  # noqa: BLE001 - metrics should not break request
            pass
        return response
