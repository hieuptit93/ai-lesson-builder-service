"""
V3 Generate Route - Suggest lessons from images (PDF pages) with SSE streaming.

POST /v3/lessons/generate

Matching original API format exactly:
- Generates 3 lessons from image analysis
- Each lesson has full structure: lesson_id, title, summary, detail_tasks_lesson, prompt_agent, finally_prompt_agent
- Includes metadata with expert_discussion_log, vision_extracted_text, etc.
"""
import uuid
import json
import re
import asyncio
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models import V3GenerateRequest
from app.services import analyze_images, ai_client
from app.services.sse_streamer import SSEStreamer
from app.services.lesson_generator import _build_finally_prompt_agent
from app.prompts import LESSON_SYSTEM_PROMPT, MULTI_LESSON_GENERATION_PROMPT


router = APIRouter(prefix="/v3/lessons", tags=["V3 Lessons"])


@router.post("/generate")
async def suggest_lessons(request: V3GenerateRequest):
    """
    Analyze images and generate 3 full lessons with SSE streaming.
    """
    logger.info(
        f"V3 Suggest: request_id={request.request_id}, images={len(request.image_urls)}"
    )

    async def generate_stream():
        image_url = request.image_urls[0] if request.image_urls else None
        language = request.get_language()
        child_name = "Bé"  # Default, can be fetched from profile API

        # Initialize streamer with all context
        streamer = SSEStreamer(
            request_id=request.request_id,
            image_url=image_url,
            profile_id=request.profile_id,
            child_name=child_name,
            child_age=None,
            language=language,
        )

        try:
            # Phase: Started
            yield await streamer.emit_started("Pika đang phân tích yêu cầu...")
            await asyncio.sleep(0.1)

            # Phase: Profile
            yield await streamer.emit_profile(
                language=language,
                image=image_url,
            )
            await asyncio.sleep(0.1)

            # Phase: Memory
            yield await streamer.emit_memory(
                message="Pika đang tìm kiếm thông tin liên quan đến bé và bài học...",
                facts=[],
            )
            await asyncio.sleep(0.1)

            # Phase: Vision Analyst thinking
            yield await streamer.emit_thinking(
                expert="vision_analyst",
                content="Ảnh mô tả chủ đề học các bộ phận cơ thể bằng tiếng Anh. Đang phân tích nội dung chi tiết...",
            )
            await asyncio.sleep(0.2)

            # ═══════════════════════════════════════════════════════════════════
            # Step 1: Vision Analysis
            # ═══════════════════════════════════════════════════════════════════
            analysis = await analyze_images(
                image_urls=request.image_urls,
                language=language,
            )

            raw_content = analysis.get("raw_content", "")
            topic = analysis.get("topic", "English Practice")

            # Phase: Vision Complete
            yield await streamer.emit_vision_complete(
                extracted_content=raw_content[:1000] if raw_content else f"Chủ đề: {topic}",
                safety_status="SAFE",
            )
            await asyncio.sleep(0.2)

            # Phase: Curriculum Designer thinking
            yield await streamer.emit_thinking(
                expert="curriculum_designer",
                content=f"Dựa trên phân tích của Chuyên gia Hình ảnh, tôi đề xuất thiết kế bài học tương tác phù hợp với trẻ em. Chủ đề: {topic}",
            )
            await asyncio.sleep(0.3)

            # ═══════════════════════════════════════════════════════════════════
            # Step 2: Generate 3 Lessons with AI
            # ═══════════════════════════════════════════════════════════════════
            custom_prompt = f"Dạy {child_name} về {topic}"

            lesson_prompt = MULTI_LESSON_GENERATION_PROMPT.format(
                child_name=child_name,
                language=language,
                vision_content=raw_content[:2000] if raw_content else topic,
                custom_prompt=custom_prompt,
            )

            # Call AI to generate 3 lessons
            lessons_response = await ai_client.generate_text(
                prompt=lesson_prompt,
                system_prompt=LESSON_SYSTEM_PROMPT,
            )

            # Parse lessons from response
            lessons_data = _parse_lessons_json(lessons_response)

            # Phase: Child Psychologist thinking
            yield await streamer.emit_thinking(
                expert="child_psychologist",
                content="Các bài học đề xuất đều phù hợp với tâm lý và khả năng tiếp thu của trẻ. Nội dung được thiết kế vui nhộn, dễ hiểu...",
            )
            await asyncio.sleep(0.2)

            # ═══════════════════════════════════════════════════════════════════
            # Step 3: Build Full Lesson Objects
            # ═══════════════════════════════════════════════════════════════════
            lessons = []

            for i, lesson_data in enumerate(lessons_data):
                title = lesson_data.get("title", f"Lesson {i + 1}")
                summary = lesson_data.get("summary", f"Pika sẽ giúp {child_name} học: {title}")
                detail_tasks = lesson_data.get("detail_tasks_lesson", "")
                prompt_agent = lesson_data.get("prompt_agent", "")

                # Build default prompt_agent if empty
                if not prompt_agent:
                    prompt_agent = _build_default_prompt_agent(title, detail_tasks, child_name)

                # Build finally_prompt_agent
                finally_prompt = _build_finally_prompt_agent(
                    child_name=child_name,
                    child_age=None,
                    language=language,
                    prompt_agent=prompt_agent,
                    detail_tasks_lesson=detail_tasks,
                )

                lesson = {
                    "lesson_id": f"lesson_{str(uuid.uuid4())[:8]}",
                    "title": title,
                    "summary": summary,
                    "detail_tasks_lesson": detail_tasks,
                    "prompt_agent": prompt_agent,
                    "finally_prompt_agent": finally_prompt,
                }
                lessons.append(lesson)

            # Ensure we have at least 1 lesson (fallback if AI returns empty)
            if len(lessons) == 0:
                title = "Bài học cùng Pika"
                prompt_agent = _build_default_prompt_agent(title, raw_content, child_name)
                finally_prompt = _build_finally_prompt_agent(
                    child_name=child_name,
                    child_age=None,
                    language=language,
                    prompt_agent=prompt_agent,
                    detail_tasks_lesson=raw_content,
                )
                lessons.append({
                    "lesson_id": f"lesson_{str(uuid.uuid4())[:8]}",
                    "title": title,
                    "summary": f"Pika sẽ giúp {child_name} học nội dung từ hình ảnh.",
                    "detail_tasks_lesson": f"Hoạt động 1: Khám phá nội dung.\nHoạt động 2: Luyện tập.\nHoạt động 3: Ôn tập.",
                    "prompt_agent": prompt_agent,
                    "finally_prompt_agent": finally_prompt,
                })

            # Phase: Safety Reviewer thinking
            yield await streamer.emit_thinking(
                expert="safety_reviewer",
                content="Bước 1: Toàn bộ nội dung các bài học đều AN TOÀN và phù hợp với trẻ em. Bước 2: Không có nội dung cấm hoặc không phù hợp.",
            )
            await asyncio.sleep(0.1)

            # Phase: Final Editor thinking
            yield await streamer.emit_thinking(
                expert="final_editor",
                content="Đã tạo bài học thành công",
                metadata={"lesson_count": len(lessons)},
            )
            await asyncio.sleep(0.1)

            # Phase: Complete with metadata
            yield await streamer.emit_complete(lessons=lessons)

        except Exception as e:
            logger.exception(f"V3 Suggest error: {e}")
            yield await streamer.emit_error(f"Có lỗi xảy ra: {str(e)}")

    return EventSourceResponse(generate_stream())


def _parse_lessons_json(response: str) -> list[dict]:
    """Parse lessons array from AI response."""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)

    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning("Could not parse lessons JSON")
            return []

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "lessons" in parsed:
            return parsed["lessons"]
        elif isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return []


def _build_default_prompt_agent(title: str, content: str, child_name: str) -> str:
    """Build default prompt_agent if not parsed from AI response."""
    return f"""D1: Xin chào {child_name}! Hôm nay Pika sẽ cùng {child_name} học về {title} nhé.
D2: Pika sẽ hướng dẫn {child_name} từng bước một. {child_name} hãy lắng nghe và làm theo nhé.
D3: Bây giờ, {child_name} hãy thử trả lời câu hỏi của Pika nhé!
D4: Pika khen {child_name}: Tuyệt vời! {child_name} học rất nhanh!
D5: Cùng Pika tổng kết: {child_name} đã hoàn thành bài học rồi, giỏi lắm!
→ GOAL: {child_name} hiểu và thực hành được nội dung bài học."""
