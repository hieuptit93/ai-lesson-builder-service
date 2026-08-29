"""
V3 Generate Artifact Route - Generate full lessons from suggestions.

POST /v3/lessons/generate_artifact
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models import V3GenerateArtifactRequest
from app.services.sse_streamer import stream_artifact_generation
from app.services.lesson_generator import _generate_single_artifact


router = APIRouter(prefix="/v3/lessons", tags=["V3 Lessons"])


@router.post("/generate_artifact")
async def generate_artifacts(request: V3GenerateArtifactRequest):
    """
    Generate full lesson artifacts from suggestions.
    Returns complete lessons with prompt_agent and finally_prompt_agent.
    """
    logger.info(
        f"V3 Artifact: request_id={request.request_id}, lessons={len(request.lessons)}"
    )

    # Default child name - can be fetched from profile
    child_name = "Bé"
    language = "vi"  # Default language

    if request.stream:
        # SSE Streaming response
        async def generate():
            async for event in stream_artifact_generation(
                suggestions=request.lessons,
                generate_func=_generate_single_artifact,
                request_id=request.request_id,
                profile_id=request.profile_id,
                child_name=child_name,
                language=language,
            ):
                yield event

        return EventSourceResponse(generate())

    else:
        # Non-streaming fallback
        try:
            lessons = []
            for suggestion in request.lessons:
                lesson = await _generate_single_artifact(
                    suggestion=suggestion,
                    request_id=request.request_id,
                    child_name=child_name,
                    language=language,
                )
                lessons.append(lesson)

            return JSONResponse(
                status_code=200,
                content={
                    "status": 200,
                    "data": {"lessons": lessons},
                },
            )

        except Exception as e:
            logger.exception(f"V3 Artifact error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": 500,
                    "message": str(e),
                },
            )
