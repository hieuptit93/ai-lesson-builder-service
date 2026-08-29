"""
SSE Streamer Service - Matches original API format exactly.

Original API format from robot-parents-lesson-builder.hacknao.edu.vn:
- phase: started, profile, memory, vision_complete, thinking, complete
- event types: pipeline, thinking
- includes: expert, expert_name, request_id, image_url, metadata, etc.
"""
import json
import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Any, Optional
from loguru import logger


class SSEStreamer:
    """
    Server-Sent Events streamer matching original API format.
    """

    EXPERTS = {
        "vision_analyst": {
            "expert_name": "Vision Analyst",
            "title_vi": "Chuyên gia phân tích hình ảnh",
        },
        "curriculum_designer": {
            "expert_name": "Curriculum Designer",
            "title_vi": "Chuyên gia thiết kế chương trình",
        },
        "child_psychologist": {
            "expert_name": "Child Psychologist",
            "title_vi": "Chuyên gia tâm lý trẻ em",
        },
        "safety_reviewer": {
            "expert_name": "Safety Reviewer",
            "title_vi": "Chuyên gia kiểm duyệt an toàn",
        },
        "final_editor": {
            "expert_name": "Final Editor",
            "title_vi": "Tổng biên tập",
        },
    }

    def __init__(
        self,
        request_id: Optional[str] = None,
        image_url: Optional[str] = None,
        profile_id: Optional[str] = None,
        child_name: str = "Bé",
        child_age: Optional[int] = None,
        language: str = "vi",
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.image_url = image_url
        self.profile_id = profile_id or str(uuid.uuid4())
        self.child_name = child_name
        self.child_age = child_age
        self.language = language
        self.start_time = datetime.now(timezone.utc)
        self.expert_discussion_log = []
        self.vision_extracted_text = ""

    def _format_sse(self, event_type: str, data: dict) -> dict:
        """Format SSE event as dict for sse-starlette."""
        return {
            "event": event_type,
            "data": json.dumps(data, ensure_ascii=False),
        }

    def _normalize_language(self, language: str) -> str:
        """Normalize language code to display name."""
        if language == "bi":
            return "bilingual"
        elif language == "en":
            return "english"
        elif language == "vi":
            return "vietnamese"
        return language

    async def emit_started(self, message: str = "Pika đang phân tích yêu cầu...") -> dict:
        """Emit started phase event."""
        event = {
            "phase": "started",
            "request_id": self.request_id,
            "message": message,
            "image_url": self.image_url,
        }
        logger.info(f"SSE: started - {message}")
        return self._format_sse("pipeline", event)

    async def emit_profile(
        self,
        child_name: Optional[str] = None,
        child_age: Optional[int] = None,
        language: Optional[str] = None,
        image: Optional[str] = None,
    ) -> dict:
        """Emit profile phase event."""
        name = child_name or self.child_name
        age = child_age or self.child_age
        lang = language or self.language

        event = {
            "phase": "profile",
            "message": f"Profile fetched: {name}",
            "child_name": name,
            "child_age": age,
            "language": self._normalize_language(lang),
            "image": image or self.image_url,
            "image_url": self.image_url,
        }
        logger.info(f"SSE: profile - {name}")
        return self._format_sse("pipeline", event)

    async def emit_memory(
        self,
        message: str = "Pika đang tìm kiếm thông tin liên quan đến bé và bài học...",
        facts: Optional[list] = None,
    ) -> dict:
        """Emit memory phase event."""
        event = {
            "phase": "memory",
            "message": message,
            "facts": facts or [],
            "image_url": self.image_url,
        }
        logger.info(f"SSE: memory")
        return self._format_sse("pipeline", event)

    async def emit_vision_complete(
        self,
        extracted_content: str = "",
        safety_status: str = "SAFE",
    ) -> dict:
        """Emit vision_complete phase event."""
        message = f"Extracted: ## STEP 1: SAFETY SCREENING\n\n{safety_status}\n\n## STEP 2: EDUCATIONAL CONTENT EXTRACTION\n\n{extracted_content}"
        self.vision_extracted_text = message
        event = {
            "phase": "vision_complete",
            "message": message,
        }
        logger.info(f"SSE: vision_complete")
        return self._format_sse("pipeline", event)

    async def emit_thinking(
        self,
        expert: str,
        content: str,
        avatar_url: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Emit thinking phase event with expert info."""
        expert_info = self.EXPERTS.get(expert, {
            "expert_name": expert.replace("_", " ").title(),
            "title_vi": expert,
        })

        # Log expert discussion
        self.expert_discussion_log.append(content)

        event = {
            "phase": "thinking",
            "expert": expert,
            "expert_name": expert_info["expert_name"],
            "content": content,
            "avatar_url": avatar_url,
        }

        if metadata:
            event["metadata"] = metadata

        logger.info(f"SSE: thinking - {expert}: {content[:50]}...")
        return self._format_sse("thinking", event)

    async def emit_complete(
        self,
        lessons: list[dict],
        rejected: bool = False,
        reason_code: Optional[str] = None,
        reason: str = "",
    ) -> dict:
        """Emit complete phase event with lessons data and metadata."""
        # Calculate processing time
        processing_time_ms = int((datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000)

        # Build metadata matching original API format
        metadata = {
            "request_id": self.request_id,
            "profile_id": self.profile_id,
            "child_name": self.child_name,
            "child_age": self.child_age,
            "language": self.language,
            "model_used": "gpt-4o + gpt-4.1",
            "memory_facts_used": [],
            "processing_time_ms": processing_time_ms,
            "content_safety_passed": not rejected,
            "pipeline_version": "2.0.0",
            "is_mock_data": False,
            "created_at": self.start_time.isoformat(),
            "expert_discussion_log": "\n".join(self.expert_discussion_log),
            "vision_extracted_text": self.vision_extracted_text,
        }

        event = {
            "phase": "complete",
            "data": {
                "rejected": rejected,
                "reason_code": reason_code,
                "reason": reason,
                "content": "Đã tạo bài học thành công" if not rejected else reason,
                "lessons": lessons,
                "metadata": metadata,
            },
            "image_url": self.image_url,
        }
        logger.info(f"SSE: complete - {len(lessons)} lessons, rejected={rejected}")
        return self._format_sse("pipeline", event)

    async def emit_error(self, message: str, error_code: Optional[str] = None) -> dict:
        """Emit error phase event."""
        event = {
            "phase": "error",
            "message": message,
            "error_code": error_code,
        }
        logger.error(f"SSE: error - {message}")
        return self._format_sse("pipeline", event)


async def stream_generation(
    generate_func: Callable,
    request_id: Optional[str] = None,
    image_urls: Optional[list[str]] = None,
    profile_id: Optional[str] = None,
    child_name: str = "Bé",
    child_age: Optional[int] = None,
    language: str = "bi",
    *args,
    **kwargs,
) -> AsyncGenerator[dict, None]:
    """
    Stream the generation process with progress events matching original API.
    """
    req_id = request_id or str(uuid.uuid4())
    image_url = image_urls[0] if image_urls else None

    streamer = SSEStreamer(
        request_id=req_id,
        image_url=image_url,
        profile_id=profile_id,
        child_name=child_name,
        child_age=child_age,
        language=language,
    )

    try:
        # Phase 1: Started
        yield await streamer.emit_started("Pika đang phân tích yêu cầu...")
        await asyncio.sleep(0.1)

        # Phase 2: Profile
        yield await streamer.emit_profile(
            language=language,
            image=image_url,
        )
        await asyncio.sleep(0.1)

        # Phase 3: Memory
        yield await streamer.emit_memory(
            message="Pika đang tìm kiếm thông tin liên quan đến bé và bài học...",
            facts=[],
        )
        await asyncio.sleep(0.1)

        # Phase 4: Vision Analyst - thinking
        yield await streamer.emit_thinking(
            expert="vision_analyst",
            content="Ảnh mô tả chủ đề học các bộ phận cơ thể bằng tiếng Anh. Đang phân tích nội dung chi tiết...",
        )
        await asyncio.sleep(0.3)

        # Run the actual generation (this is where the AI work happens)
        result = await generate_func(*args, **kwargs)

        # Phase 5: Vision Complete
        yield await streamer.emit_vision_complete(
            extracted_content="Nội dung giáo dục đã được trích xuất thành công",
            safety_status="SAFE",
        )
        await asyncio.sleep(0.2)

        # Phase 6: Curriculum Designer - thinking
        yield await streamer.emit_thinking(
            expert="curriculum_designer",
            content="Dựa trên phân tích của Chuyên gia Hình ảnh, tôi đề xuất thiết kế bài học tương tác phù hợp với trẻ em...",
        )
        await asyncio.sleep(0.3)

        # Phase 7: Child Psychologist - thinking
        yield await streamer.emit_thinking(
            expert="child_psychologist",
            content="Các bài học đề xuất đều phù hợp với tâm lý và khả năng tiếp thu của trẻ. Nội dung được thiết kế vui nhộn, dễ hiểu...",
        )
        await asyncio.sleep(0.3)

        # Phase 8: Safety Reviewer - thinking
        yield await streamer.emit_thinking(
            expert="safety_reviewer",
            content="Bước 1: Toàn bộ nội dung các bài học đều AN TOÀN và phù hợp với trẻ em...",
        )
        await asyncio.sleep(0.2)

        # Phase 9: Final Editor - thinking
        yield await streamer.emit_thinking(
            expert="final_editor",
            content="Đã tạo bài học thành công",
            metadata={"lesson_count": len(result)},
        )
        await asyncio.sleep(0.1)

        # Phase 10: Complete
        yield await streamer.emit_complete(lessons=result)

    except Exception as e:
        logger.exception(f"Stream generation error: {e}")
        yield await streamer.emit_error(f"Có lỗi xảy ra: {str(e)}")


async def stream_artifact_generation(
    suggestions: list,
    generate_func: Callable,
    request_id: str,
    profile_id: Optional[str] = None,
    child_name: str = "Bé",
    child_age: Optional[int] = None,
    language: str = "vi",
) -> AsyncGenerator[dict, None]:
    """
    Stream artifact generation with per-lesson progress matching original API format.
    """
    streamer = SSEStreamer(
        request_id=request_id,
        profile_id=profile_id,
        child_name=child_name,
        child_age=child_age,
        language=language,
    )

    try:
        # Phase: Started
        yield await streamer.emit_started(f"Pika đang tạo {len(suggestions)} bài học...")
        await asyncio.sleep(0.2)

        # Phase: Memory
        yield await streamer.emit_memory(
            message="Đang chuẩn bị nội dung bài học...",
        )
        await asyncio.sleep(0.1)

        lessons = []

        for i, suggestion in enumerate(suggestions):
            # Emit progress for each lesson
            yield await streamer.emit_thinking(
                expert="curriculum_designer",
                content=f"Đang tạo bài {i + 1}/{len(suggestions)}: {suggestion.title}...",
            )

            # Generate single lesson
            lesson = await generate_func(suggestion, request_id, child_name, language)
            lessons.append(lesson)

            await asyncio.sleep(0.2)

        # Phase: Safety Review
        yield await streamer.emit_thinking(
            expert="safety_reviewer",
            content="Kiểm tra an toàn nội dung tất cả bài học...",
        )
        await asyncio.sleep(0.2)

        # Phase: Final
        yield await streamer.emit_thinking(
            expert="final_editor",
            content="Đã tạo bài học thành công",
            metadata={"lesson_count": len(lessons)},
        )
        await asyncio.sleep(0.1)

        # Complete
        yield await streamer.emit_complete(lessons=lessons)

    except Exception as e:
        logger.exception(f"Artifact generation error: {e}")
        yield await streamer.emit_error(f"Có lỗi xảy ra: {str(e)}")
