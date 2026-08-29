"""API Key Authentication Middleware.

Validates X-API-Key header for service-to-service authentication.
Supports two modes:
- Warning mode (X_API_KEY_ENABLED=false): Allow requests but alert on missing/invalid key
- Strict mode (X_API_KEY_ENABLED=true): Reject requests with 401 if key is missing/invalid
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import structlog

from app.core.config import Settings

logger = structlog.get_logger()

# Endpoints that don't require API key authentication
SKIP_PATHS = {
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/v1/health",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key from X-API-Key header."""

    X_API_KEY_HEADER = "X-API-Key"

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip authentication for certain paths
        if request.url.path in SKIP_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get(self.X_API_KEY_HEADER)
        is_valid = api_key == self._settings.x_api_key

        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = str(request.url.path)

        if self._settings.x_api_key_enabled:
            # Strict mode: Reject if key is missing or invalid
            if not is_valid:
                logger.warning(
                    "api_key.rejected",
                    client_ip=client_ip,
                    method=method,
                    path=path,
                    reason="invalid_or_missing_key",
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: Invalid or missing API key"},
                )
        else:
            # Warning mode: Allow through but alert
            if not api_key or not is_valid:
                logger.warning(
                    "api_key.missing_or_invalid",
                    client_ip=client_ip,
                    method=method,
                    path=path,
                    reason="warning_mode_allowing_request",
                )

        return await call_next(request)
