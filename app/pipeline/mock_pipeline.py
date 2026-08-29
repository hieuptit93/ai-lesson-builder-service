"""MockPipeline: returns mock lesson data without calling real LLMs.

Supports both sync (JSON) and async SSE streaming modes.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from uuid import uuid4

from app.api.v1.schemas.lesson_request import GenerateLessonRequest, RegenerateLessonRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Expert discussion logs (multi-persona thinking mock)
# ---------------------------------------------------------------------------

_EXPERT_LOGS = {
    "english": {
        "vision_analyst": '[A] Vision Analyst: Colors and Shapes vocabulary page. 4 colors + 3 shapes detected. Beginner level.',
        "curriculum_designer": '[B] Curriculum Designer: English Template - word_echo -> story_context -> quick_quiz -> fun_fact_stars.',
        "child_psychologist": '[C] Child Psychologist: Child is 5yo - use real-world analogies (pizza=triangle, ball=circle). From memory: likes butterflies -> use butterfly color fact in wrap_up.',
        "safety_reviewer": '[D] Safety Reviewer: All facts accurate. Age-appropriate.',
        "final_editor": '[E] Final Editor: Applied butterfly hook from C. Generating final JSON...',
    },
    "math": {
        "vision_analyst": '[A] Vision Analyst: No image provided. Prompt-based: addition/subtraction within 20. Beginner level for grade 1.',
        "curriculum_designer": '[B] Curriculum Designer: Math Template - curiosity_spark -> explain_discuss -> mental_math_drill -> word_problems -> fun_fact_stars. 5 drill questions + 1 word problem.',
        "child_psychologist": '[C] Child Psychologist: Child is 6yo grade 1. Use candy/marble analogies. Start with range 1-10, then 10-20. If wrong 2x, suggest counting on fingers.',
        "safety_reviewer": '[D] Safety Reviewer: All math facts correct. No issues.',
        "final_editor": '[E] Final Editor: Applied C\'s progressive difficulty. Generating final JSON...',
    },
    "science": {
        "vision_analyst": '[A] Vision Analyst: Human body parts diagram. Key vocab: heart, lungs, stomach + 3 more. Intermediate difficulty.',
        "curriculum_designer": '[B] Curriculum Designer: Science Template - curiosity_spark -> explain_discuss -> true_or_false -> fun_fact_stars. Bilingual: intro in Vietnamese, terms in English.',
        "child_psychologist": '[C] Child Psychologist: Child is 7yo at bilingual school. From memory: learned animals last month -> connect \'body systems work like animal teamwork\'. Use touch-based activities.',
        "safety_reviewer": '[D] Safety Reviewer: Anatomy facts correct. B wrote \'heart helps you breathe\' in quiz - corrected to false answer with lung explanation. Bilingual balance OK.',
        "final_editor": '[E] Final Editor: Applied D\'s heart/lung correction. Added C\'s animal connection. Generating final JSON...',
    },
}


# ---------------------------------------------------------------------------
# Full mock lesson data per subject
# ---------------------------------------------------------------------------

def _english_lesson_data(req: GenerateLessonRequest, request_id: str) -> dict:
    child_name = req.parent_config.child_name or "Child"
    return {
        "lesson_id": f"lesson-eng-{uuid4().hex[:6]}",
        "topic": "Colors and Shapes",
        "subject": "english",
        "language": req.parent_config.language,
        "difficulty": "beginner",
        "estimated_duration_min": 10,
        "extracted_content": {
            "raw_text": "Colors: Red, Blue, Green, Yellow. Shapes: Circle, Square, Triangle",
            "topic_detected": "Colors and Shapes",
            "subject_detected": "english",
            "confidence_score": 0.97,
        },
        "vocabulary": [
            {"word": "circle", "meaning": "A round shape, like a ball", "example": "The sun looks like a big yellow circle!", "pronunciation_guide": "SUR-kul"},
            {"word": "triangle", "meaning": "A shape with three sides", "example": "A slice of pizza looks like a triangle!", "pronunciation_guide": "TRY-ang-gul"},
        ],
        "concepts": [
            {"name": "Shapes Around Us", "explanation": "Shapes are everywhere! A door is a rectangle, a wheel is a circle.", "check_question": "Can you tell me something that looks like a circle?"},
        ],
        "activities": [
            {
                "phase": "warm_up", "type": "word_echo", "title": "Say the Colors!",
                "prompt_template": f"Hi {child_name}! Let's practice colors! I'll say a color and you repeat after me. Ready? RED!",
                "duration_min": 2, "questions": None, "adaptive_rules": {"if_wrong": "repeat_slowly", "if_correct": "praise_and_continue", "if_no_response": "give_hint", "max_retries": 2},
            },
            {
                "phase": "present", "type": "story_context", "title": "The Colorful Garden",
                "prompt_template": f"Once upon a time, there was a garden full of colors. The roses were RED, the sky was BLUE, and the grass was GREEN. {child_name}, what color is the sun?",
                "duration_min": 3, "questions": None, "adaptive_rules": {"if_wrong": "give_hint", "if_correct": "next_question", "if_no_response": "give_hint", "max_retries": 2},
            },
        ],
        "metadata": {
            "request_id": request_id,
            "model_used": "gpt-4o-vision",
            "memory_facts_used": [f"{child_name} is 5 years old", f"{child_name} likes butterflies"],
            "processing_time_ms": 6200,
            "content_safety_passed": True,
            "pipeline_version": "2.0.0-mock",
            "created_at": _now_iso(),
        },
    }


def _build_response_regenerate(req: RegenerateLessonRequest, request_id: str) -> dict:
    subject = _resolve_subject(req)
    builder = _BUILDERS.get(subject, _english_lesson_data)
    data = builder(req, request_id)

    lesson = {
        "lesson_id": "lesson_regen",
        "title": data["topic"] if "topic" in data else "English Lesson",
        "summary": "Mock summary for regeneration",
        "detail_tasks_lesson": "Activity 1: Pika describes...\nActivity 2: Child guesses...",
        "prompt_agent": "D1: Step 1\nD2: Step 2\n-> GOAL: Done",
    }

    return {
        "status": "success",
        "lessons": [lesson],
        "metadata": {
            "request_id": request_id,
            "parent_lesson_id": req.lessons_needing_regeneration.lesson_id,
            "processing_time_ms": 1200,
            "content_safety_passed": True,
            "pipeline_version": "2.0.0-regen",
            "created_at": _now_iso(),
        },
    }


def _math_lesson_data(req: GenerateLessonRequest, request_id: str) -> dict:
    child_name = req.parent_config.child_name or "Child"
    return {
        "lesson_id": f"lesson-math-{uuid4().hex[:6]}",
        "topic": "Addition and Subtraction within 20",
        "subject": "math",
        "language": req.parent_config.language,
        "difficulty": "beginner",
        "estimated_duration_min": 10,
        "extracted_content": {
            "raw_text": "",
            "topic_detected": "Basic addition subtraction",
            "subject_detected": "math",
            "confidence_score": 1.0,
        },
        "vocabulary": [
            {"word": "add", "meaning": "Combine numbers together", "example": "3 add 2 equals 5", "pronunciation_guide": ""},
            {"word": "subtract", "meaning": "Take away a number from another", "example": "5 subtract 2 equals 3", "pronunciation_guide": ""},
        ],
        "concepts": [
            {"name": "Addition within 20", "explanation": "When adding two numbers, we combine them. For example: you have 5 apples, mom gives you 3 more, now you have 8 apples.", "check_question": "5 add 3 equals how much?"},
        ],
        "activities": [
            {
                "phase": "warm_up", "type": "curiosity_spark", "title": "Quick counting!",
                "prompt_template": f"Hi {child_name}! Today we play with numbers! Count from 1 to 10 for Pika!",
                "duration_min": 2, "questions": None, "adaptive_rules": {"if_wrong": "simplify", "if_correct": "praise_and_continue", "if_no_response": "count_together", "max_retries": 2},
            },
        ],
        "metadata": {
            "request_id": request_id,
            "model_used": "gpt-4.1",
            "memory_facts_used": [f"{child_name} is 6 years old", f"{child_name} is learning grade 1 math"],
            "processing_time_ms": 4300,
            "content_safety_passed": True,
            "pipeline_version": "2.0.0-mock",
            "created_at": _now_iso(),
        },
    }


def _science_lesson_data(req: GenerateLessonRequest, request_id: str) -> dict:
    child_name = req.parent_config.child_name or "Child"
    return {
        "lesson_id": f"lesson-sci-{uuid4().hex[:6]}",
        "topic": "Parts of the Human Body",
        "subject": "science",
        "language": req.parent_config.language,
        "difficulty": "intermediate",
        "estimated_duration_min": 12,
        "extracted_content": {
            "raw_text": "The Human Body: Head, Arms, Legs, Heart, Lungs, Stomach...",
            "topic_detected": "Human Body Parts",
            "subject_detected": "science",
            "confidence_score": 0.93,
        },
        "vocabulary": [
            {"word": "heart", "meaning": "The part inside your body that pumps blood", "example": "Your heart beats about 100,000 times every day!", "pronunciation_guide": "HART"},
            {"word": "lungs", "meaning": "The parts inside your chest that help you breathe", "example": "When you breathe in, air goes into your lungs.", "pronunciation_guide": "LUNGZ"},
        ],
        "concepts": [
            {
                "name": "Body Systems",
                "explanation": "Our body has many parts that work together like a team! Your heart pumps blood, your lungs help you breathe, and your stomach helps digest food.",
                "check_question": "What does your heart do?",
            },
        ],
        "activities": [
            {
                "phase": "warm_up", "type": "curiosity_spark", "title": "Point and Say!",
                "prompt_template": f"Hey {child_name}! Let's play a game! Touch your head! Now touch your nose! What is this called in English?",
                "duration_min": 2, "questions": None, "adaptive_rules": {"if_wrong": "say_both_languages", "if_correct": "praise_and_continue", "if_no_response": "give_hint", "max_retries": 2},
            },
        ],
        "metadata": {
            "request_id": request_id,
            "model_used": "gpt-4o-vision + gpt-4.1",
            "memory_facts_used": [f"{child_name} is 7 years old", f"{child_name} studies at bilingual school", f"{child_name} learned about animals last month"],
            "processing_time_ms": 9100,
            "content_safety_passed": True,
            "pipeline_version": "2.0.0-mock",
            "created_at": _now_iso(),
        },
    }


# Subject -> builder mapping
_BUILDERS = {
    "english": _english_lesson_data,
    "math": _math_lesson_data,
    "science": _science_lesson_data,
}


def _resolve_subject(req: GenerateLessonRequest) -> str:
    return req.parent_config.subject or "english"


class MockPipeline:
    """Pipeline that returns pre-built mock data for BE integration testing."""

    async def generate(self, request: GenerateLessonRequest, *, request_id: str) -> dict:
        subject = _resolve_subject(request)
        builder = _BUILDERS.get(subject, _english_lesson_data)
        data = builder(request, request_id)
        return {
            "request_id": request_id,
            "status": "success",
            "data": data,
        }

    async def generate_v3(self, request: GenerateLessonRequest, *, request_id: str) -> dict:
        return await self.generate(request, request_id=request_id)

    async def generate_regenerate(self, request: RegenerateLessonRequest, *, request_id: str) -> dict:
        data = _build_response_regenerate(request, request_id)
        return {
            "request_id": request_id,
            "status": "success",
            "data": data,
        }

    async def generate_artifact_v3(self, request, *, request_id: str) -> dict:
        return {
            "request_id": request_id,
            "status": "success",
            "data": {
                "rejected": False,
                "reason_code": None,
                "reason": "",
                "content": "Created lessons successfully (mock)",
                "lessons": [],
                "metadata": {
                    "processing_time_ms": 100,
                    "pipeline_version": "2.0.0-mock",
                    "created_at": _now_iso(),
                },
            },
        }

    async def stream_generate(
        self,
        request: GenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.8,
    ) -> AsyncGenerator[str, None]:
        subject = _resolve_subject(request)
        experts = _EXPERT_LOGS.get(subject, _EXPERT_LOGS["english"])

        yield _sse_event({"phase": "started", "request_id": request_id, "message": "Pipeline started"})
        await asyncio.sleep(delay * 0.5)

        yield _sse_event({"phase": "memory", "message": "Fetching child memory from Mem0..."})
        await asyncio.sleep(delay)

        if request.image_urls:
            yield _sse_event({"phase": "vision", "message": f"Extracting content from {len(request.image_urls)} image(s)..."})
            await asyncio.sleep(delay)

            yield _sse_event({"phase": "safety_check", "message": "Checking image safety..."})
            await asyncio.sleep(delay * 0.5)

        expert_keys = ["vision_analyst", "curriculum_designer", "child_psychologist", "safety_reviewer", "final_editor"]
        for key in expert_keys:
            yield _sse_event({"phase": "thinking", "expert": key, "content": experts[key]})
            await asyncio.sleep(delay)

        builder = _BUILDERS.get(subject, _english_lesson_data)
        data = builder(request, request_id)

        yield _sse_event({"phase": "complete", "data": data})
        yield "data: [DONE]\n\n"

    async def stream_generate_v3(
        self,
        request: GenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.8,
    ) -> AsyncGenerator[str, None]:
        async for event in self.stream_generate(request, request_id=request_id, delay=delay):
            yield event

    async def stream_regenerate(
        self,
        request: RegenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.8,
    ) -> AsyncGenerator[str, None]:
        yield _sse_event({"phase": "started", "request_id": request_id, "message": "Pika is regenerating lesson..."})
        await asyncio.sleep(delay)

        yield _sse_event({"phase": "thinking", "expert": "final_editor", "content": "Generating mock regeneration..."})
        await asyncio.sleep(delay)

        data = _build_response_regenerate(request, request_id)
        yield _sse_event({"phase": "complete", "data": data})
        yield "data: [DONE]\n\n"

    async def stream_artifact_v3(
        self,
        request,
        *,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        yield _sse_event({"phase": "started", "request_id": request_id, "message": "Creating artifacts (mock)..."})
        await asyncio.sleep(0.5)

        yield _sse_event({"phase": "complete", "data": {"lessons": [], "metadata": {"pipeline_version": "2.0.0-mock"}}})
        yield "data: [DONE]\n\n"


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
