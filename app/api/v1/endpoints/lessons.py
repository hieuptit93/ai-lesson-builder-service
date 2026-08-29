import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_pipeline, get_settings
from app.api.v1.schemas.lesson_request import GenerateLessonRequest, RegenerateLessonRequest
from app.api.v1.schemas.lesson_response import GenerateLessonResponse
from app.core.config import Settings
from app.pipeline.lesson_pipeline import LessonPipeline

router = APIRouter(prefix="/lessons", tags=["lessons"])
logger = structlog.get_logger()


@router.post(
    "/generate",
    response_model=GenerateLessonResponse,
    responses={
        422: {"description": "Unsafe content or extraction failed"},
        502: {"description": "LLM provider unavailable"},
    },
    include_in_schema=True,
)
async def generate_lesson(
    body: GenerateLessonRequest,
    request: Request,
    pipeline: LessonPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    request_id = body.request_id or request.headers.get("X-Request-ID") or str(uuid4())
    start_time = time.monotonic()

    # Get subject from optional_parent_config if present
    parent_config = body.optional_parent_config
    subject = parent_config.subject if parent_config else None

    logger.info(
        "lesson_generation_started",
        request_id=request_id,
        profile_id=body.profile_id,
        subject=subject,
        stream=body.stream,
        image_count=len(body.image_urls),
    )

    if body.stream:
        # Use V3 flow (Option B): only 2 API calls (guardrail + vision)
        return StreamingResponse(
            pipeline.stream_generate_v3(body, request_id=request_id, delay=settings.stream_delay_seconds),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    # Use V3 flow (Option B): only 2 API calls (guardrail + vision)
    result = await pipeline.generate_v3(body, request_id=request_id)
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    logger.info(
        "lesson_generation_completed",
        request_id=request_id,
        profile_id=body.profile_id,
        elapsed_ms=elapsed_ms,
    )

    return GenerateLessonResponse(**result)


@router.post(
    "/regenerate",
    response_model=GenerateLessonResponse,
    responses={
        422: {"description": "Unsafe content or extraction failed"},
        502: {"description": "LLM provider unavailable"},
    },
    include_in_schema=True,
)
async def regenerate_lesson(
    body: RegenerateLessonRequest,
    request: Request,
    pipeline: LessonPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    request_id = body.request_id or request.headers.get("X-Request-ID") or str(uuid4())
    start_time = time.monotonic()

    # Get subject from optional_parent_config if present
    parent_config = body.optional_parent_config
    subject = parent_config.subject if parent_config else None

    logger.info(
        "lesson_regeneration_started",
        request_id=request_id,
        profile_id=body.profile_id,
        parent_lesson_id=body.lessons_needing_regeneration.lesson_id,
        subject=subject,
        stream=body.stream,
    )

    if body.stream:
        return StreamingResponse(
            pipeline.stream_regenerate(body, request_id=request_id, delay=settings.stream_delay_seconds),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    result = await pipeline.generate_regenerate(body, request_id=request_id)
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    logger.info(
        "lesson_regeneration_completed",
        request_id=request_id,
        profile_id=body.profile_id,
        elapsed_ms=elapsed_ms,
    )

    return GenerateLessonResponse(**result)
