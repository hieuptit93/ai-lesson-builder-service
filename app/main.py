from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import secrets
import structlog
from fastapi import Depends, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.api.middleware.api_key import APIKeyMiddleware
from app.api.middleware.correlation_id import CorrelationIdMiddleware
from app.api.middleware.error_handler import ai_service_exception_handler, generic_exception_handler
from app.api.middleware.request_logger import RequestLoggerMiddleware
from app.api.v1.router import router as v1_router
from app.api.v3.router import router as v3_router
from app.core.config import Settings
from app.core.exceptions import AIServiceError
from app.core.logging import setup_logging

logger = structlog.get_logger()

# --- Swagger UI HTTP Basic Auth ---
_swagger_basic = HTTPBasic()


def _verify_swagger_credentials(
    credentials: HTTPBasicCredentials = Depends(_swagger_basic),
) -> HTTPBasicCredentials:
    correct_username = secrets.compare_digest(credentials.username, Settings().swagger_username)
    correct_password = secrets.compare_digest(credentials.password, Settings().swagger_password)
    if not (correct_username and correct_password):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    setup_logging(settings.log_level)
    logger.info("ai_lesson_builder_service_started", version=settings.app_version, debug=settings.debug)
    yield
    logger.info("ai_lesson_builder_service_shutdown")


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="AI Lesson Builder Service",
        description="AI Lesson Generator - GPT-4o Vision + GPT-4.1 with SSE Streaming",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggerMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(APIKeyMiddleware, settings=settings)

    app.add_exception_handler(AIServiceError, ai_service_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.include_router(v1_router)
    app.include_router(v3_router)

    # --- Root and health endpoints ---
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "service": "AI Lesson Builder Service",
            "version": settings.app_version,
            "endpoints": {
                "/v1/lessons/generate": "Direct generation from image + prompt (SSE)",
                "/v1/lessons/regenerate": "Regenerate a lesson",
                "/v3/lessons/generate": "Suggest lessons from images (JSON/SSE)",
                "/v3/lessons/generate_artifact": "Generate full lessons (JSON/SSE)",
                "/health": "Health check",
                "/docs": "API documentation",
            },
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "ai-lesson-builder-service",
            "version": settings.app_version,
        }

    # --- Custom Swagger/ReDoc endpoints ---
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui(credentials: HTTPBasicCredentials = Depends(_verify_swagger_credentials)):
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
        )

    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc(credentials: HTTPBasicCredentials = Depends(_verify_swagger_credentials)):
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - ReDoc",
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
