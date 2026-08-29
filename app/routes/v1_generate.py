"""
V1 Generate Route - Direct lesson generation from image + prompt.

POST /v1/lessons/generate
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models import V1GenerateRequest
from app.services import (
    analyze_images,
    generate_lessons_from_analysis,
    stream_generation,
    select_template,
    detect_exercise_subtypes,
)


router = APIRouter(prefix="/v1/lessons", tags=["V1 Lessons"])


@router.post("/generate")
async def generate_lessons(request: V1GenerateRequest):
    """
    Generate lessons from image(s) with custom prompt.

    This is the main endpoint for "Photo to Lesson" feature.
    Supports SSE streaming for real-time progress updates.

    Request:
        - profile_id: User profile ID
        - image_urls: Array of image URLs to analyze
        - custom_prompt: User's learning request
        - optional_parent_config.language: en | vi | bi
        - stream: true for SSE streaming

    Response (SSE):
        - phase: started | thinking | complete | error
        - expert: vision_analyst | curriculum_designer | ...
        - data.lessons: Array of generated lessons (on complete)
    """
    logger.info(f"V1 Generate: profile={request.profile_id}, images={len(request.image_urls)}")

    language = request.optional_parent_config.language if request.optional_parent_config else "bi"

    if request.stream:
        # SSE Streaming response with proper format
        async def generate():
            # The actual generation logic
            async def do_generation():
                # Step 1: Analyze images
                analysis = await analyze_images(
                    image_urls=request.image_urls,
                    custom_prompt=request.custom_prompt,
                    language=language,
                )

                # Step 2: Generate lessons
                lessons = await generate_lessons_from_analysis(
                    analysis=analysis,
                    custom_prompt=request.custom_prompt,
                    language=language,
                    image_urls=request.image_urls,
                )

                return lessons

            # Stream with progress events - pass image_urls and language
            async for event in stream_generation(
                do_generation,
                request_id=request.profile_id,
                image_urls=request.image_urls,
                language=language,
            ):
                yield event

        return EventSourceResponse(generate(), media_type="text/event-stream")

    else:
        # Non-streaming response (fallback)
        try:
            analysis = await analyze_images(
                image_urls=request.image_urls,
                custom_prompt=request.custom_prompt,
                language=language,
            )

            lessons = await generate_lessons_from_analysis(
                analysis=analysis,
                custom_prompt=request.custom_prompt,
                language=language,
                image_urls=request.image_urls,
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": 200,
                    "data": {"lessons": lessons},
                },
            )

        except Exception as e:
            logger.exception(f"V1 Generate error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": 500,
                    "message": str(e),
                },
            )
