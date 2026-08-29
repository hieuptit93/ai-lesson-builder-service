import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
import structlog

from app.core.tracing import flush_langfuse

logger = structlog.get_logger()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start_time = time.perf_counter()

        # Clear and bind context variables for logging
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)

        # Calculate duration for logging
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Log request completion
        logger.info(
            "request.completed",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # For streaming responses, DO NOT flush here - pipeline will flush
        # when streaming is complete. For non-streaming, flush now.
        is_streaming = isinstance(response, StreamingResponse)

        logger.info(
            "middleware_flush_check",
            is_streaming=is_streaming,
            will_flush=not is_streaming,
            endpoint=str(request.url.path),
        )

        # Add request_id to response headers
        response.headers["X-Request-ID"] = request_id

        # Flush only for non-streaming responses
        if not is_streaming:
            flush_langfuse()

        return response
