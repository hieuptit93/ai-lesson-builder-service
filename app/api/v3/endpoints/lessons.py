import json
import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_pipeline, get_settings
from app.api.v1.schemas.lesson_request import GenerateLessonRequest
from app.api.v3.schemas.artifact_request import GenerateArtifactRequest, LessonItem
from app.core.config import Settings
from app.pipeline.lesson_pipeline import LessonPipeline

router = APIRouter(prefix="/lessons", tags=["lessons-v3"])
logger = structlog.get_logger()


@router.post(
    "/generate",
    responses={
        422: {"description": "Unsafe content or extraction failed"},
        502: {"description": "LLM provider unavailable"},
    },
)
async def generate_lesson(
    body: GenerateLessonRequest,
    request: Request,
    pipeline: LessonPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    request_id = body.request_id or request.headers.get("X-Request-ID") or str(uuid4())
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
        return StreamingResponse(
            pipeline.stream_generate_v3(body, request_id=request_id, delay=settings.stream_delay_seconds),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    start_time = time.monotonic()
    result = await pipeline.generate_v3(body, request_id=request_id)
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "lesson_generation_completed",
        request_id=request_id,
        profile_id=body.profile_id,
        elapsed_ms=elapsed_ms,
    )
    return {"request_id": request_id, **result}


@router.post(
    "/generate_artifact",
    responses={
        422: {"description": "Unsafe content or extraction failed"},
        502: {"description": "LLM provider unavailable"},
    },
)
async def generate_artifact(
    body: GenerateArtifactRequest,
    request: Request,
    pipeline: LessonPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    request_id = body.request_id or request.headers.get("X-Request-ID") or str(uuid4())

    logger.info(
        "artifact_generation_started",
        request_id=request_id,
        lesson_count=len(body.lessons),
        stream=body.stream,
    )

    if body.stream:
        return StreamingResponse(
            pipeline.stream_artifact_v3(body, request_id=request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    return await pipeline.generate_artifact_v3(body, request_id=request_id)


_MAX_FILE_SIZE = 100 * 1024  # 100KB


@router.post(
    "/generate_artifact_with_file",
    responses={
        400: {"description": "Invalid file type or empty content"},
        413: {"description": "File too large"},
        422: {"description": "Invalid lessons JSON"},
        502: {"description": "LLM provider unavailable"},
    },
)
async def generate_artifact_with_file(
    request: Request,
    file: UploadFile = File(..., description="Prompt template file (.txt) - content overrides artifact generation prompt"),
    profile_id: str = Form(...),
    lessons: str = Form(..., description="JSON array of LessonItem objects"),
    profile_api_token: str | None = Form(default=None),
    request_id: str | None = Form(default=None),
    stream: bool = Form(default=False),
    pipeline: LessonPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    if not (file.filename or "").endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted")

    raw_bytes = await file.read()
    if len(raw_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100KB limit")

    try:
        custom_prompt = raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    if not custom_prompt:
        raise HTTPException(status_code=400, detail="File content cannot be empty")

    try:
        parsed = json.loads(lessons)
        if isinstance(parsed, dict):
            parsed = parsed.get("lessons", parsed)
        lesson_items = [LessonItem(**item) for item in parsed]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid lessons JSON: {exc}")

    req_id = request_id or request.headers.get("X-Request-ID") or str(uuid4())

    body = GenerateArtifactRequest(
        request_id=req_id,
        profile_id=profile_id,
        profile_api_token=profile_api_token,
        custom_learn_prompt=custom_prompt,
        custom_talk_prompt=custom_prompt,
        stream=stream,
        lessons=lesson_items,
    )

    logger.info(
        "artifact_generation_with_file_started",
        request_id=req_id,
        lesson_count=len(lesson_items),
        prompt_length=len(custom_prompt),
        stream=stream,
    )

    if stream:
        return StreamingResponse(
            pipeline.stream_artifact_v3(body, request_id=req_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": req_id,
            },
        )

    return await pipeline.generate_artifact_v3(body, request_id=req_id)
