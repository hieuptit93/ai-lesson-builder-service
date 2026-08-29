"""
Lesson Generator Service.

Generates complete lesson artifacts matching original API format exactly.
Each lesson includes: lesson_id, title, summary, detail_tasks_lesson, prompt_agent, finally_prompt_agent
"""
import json
import re
import uuid
from typing import Optional, AsyncGenerator
from loguru import logger

from app.services.ai_client import ai_client
from app.services.template_matcher import get_template_info
from app.models import LessonSuggestion
from app.models.response import DEFAULT_FINALLY_PROMPT_TEMPLATE
from app.prompts import LESSON_SYSTEM_PROMPT, LESSON_GENERATION_PROMPT, ARTIFACT_GENERATION_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

FULL_LESSON_PROMPT = """Tạo bài học tương tác hoàn chỉnh cho trẻ em học tiếng Anh.

## THÔNG TIN ĐẦU VÀO

Tên bé: {child_name}
Chủ đề/Tiêu đề: {title}
Loại template: {template_name}
Ngôn ngữ hướng dẫn: {language}
Nội dung gốc:
```
{content}
```

## YÊU CẦU OUTPUT

Tạo bài học với cấu trúc JSON sau:

```json
{{
  "title": "Tiêu đề bài học hấp dẫn",
  "summary": "Mô tả ngắn 1-2 câu về bài học, nhắc đến tên bé",
  "detail_tasks_lesson": "Mô tả chi tiết các hoạt động trong bài học:\\nHoạt động 1: ...\\nHoạt động 2: ...\\nHoạt động 3: ...",
  "prompt_agent": "D1: Lời chào và giới thiệu bài học\\nD2: Hoạt động đầu tiên...\\nD3: ...\\nD4: ...\\nD5: Tổng kết\\n→ GOAL: Mục tiêu bài học"
}}
```

## HƯỚNG DẪN CHI TIẾT

1. **title**: Ngắn gọn, hấp dẫn, có thể nhắc tên Pika
2. **summary**: 1-2 câu mô tả, nhắc tên bé {child_name}
3. **detail_tasks_lesson**: Chia thành 3 hoạt động rõ ràng
4. **prompt_agent**: Chia thành các bước D1, D2, D3, D4, D5 và kết thúc bằng → GOAL

Trả về JSON hợp lệ, không có text thừa."""


async def generate_lessons_from_analysis(
    analysis: dict,
    custom_prompt: str,
    language: str,
    image_urls: list[str],
    request_id: Optional[str] = None,
    child_name: str = "Bé",
) -> list[dict]:
    """
    Generate complete lessons from vision analysis.
    Used by /v1/lessons/generate.
    """
    logger.info(f"Generating lessons from analysis, language={language}")

    # Build generation prompt
    prompt = _build_generation_prompt(analysis, custom_prompt, language)

    # Generate with AI
    response = await ai_client.generate_text(
        prompt=prompt,
        system_prompt=LESSON_SYSTEM_PROMPT,
    )

    # Parse lessons from response
    lessons = _parse_lessons_response(response, request_id, image_urls, child_name, language)

    logger.info(f"Generated {len(lessons)} lessons")
    return lessons


async def generate_artifacts_from_suggestions(
    suggestions: list[LessonSuggestion],
    request_id: str,
    child_name: str = "Bé",
    language: str = "vi",
) -> list[dict]:
    """
    Generate full lesson artifacts from suggestions.
    Used by /v3/lessons/generate_artifact.
    """
    logger.info(f"Generating artifacts for {len(suggestions)} suggestions")

    lessons = []

    for suggestion in suggestions:
        lesson = await _generate_single_artifact(suggestion, request_id, child_name, language)
        lessons.append(lesson)

    return lessons


async def _generate_single_artifact(
    suggestion: LessonSuggestion,
    request_id: str,
    child_name: str = "Bé",
    language: str = "vi",
) -> dict:
    """Generate a single lesson artifact matching original API format."""
    template_info = get_template_info(suggestion.template_id)
    template_name = template_info.get("name", "Bài học") if template_info else "Bài học"
    template_name_en = template_info.get("name_en", "Lesson") if template_info else "Lesson"

    # Determine language instruction
    lang_mode = "VIETNAMESE ONLY"
    if language == "en":
        lang_mode = "ENGLISH ONLY"
    elif language == "bi":
        lang_mode = "BILINGUAL (English lessons, Vietnamese instructions)"

    prompt = FULL_LESSON_PROMPT.format(
        title=suggestion.title,
        template_name=f"{template_name} ({template_name_en})",
        child_name=child_name,
        language=lang_mode,
        content=suggestion.content,
    )

    response = await ai_client.generate_text(
        prompt=prompt,
        system_prompt=LESSON_SYSTEM_PROMPT,
    )

    # Parse response
    parsed = _parse_json_from_text(response)

    # Build prompt_agent from parsed data
    prompt_agent = parsed.get("prompt_agent", "")
    if not prompt_agent:
        prompt_agent = _build_default_prompt_agent(
            title=suggestion.title,
            content=suggestion.content,
            child_name=child_name,
        )

    # Build finally_prompt_agent (full system prompt)
    finally_prompt_agent = _build_finally_prompt_agent(
        child_name=child_name,
        child_age=None,
        language=language,
        prompt_agent=prompt_agent,
        detail_tasks_lesson=parsed.get("detail_tasks_lesson", ""),
    )

    # Build full lesson object matching original API format
    lesson = {
        "lesson_id": f"lesson_{str(uuid.uuid4())[:8]}",
        "title": parsed.get("title", suggestion.title),
        "summary": parsed.get("summary", f"Pika sẽ giúp {child_name} học bài: {suggestion.title}"),
        "detail_tasks_lesson": parsed.get("detail_tasks_lesson", suggestion.content),
        "prompt_agent": prompt_agent,
        "finally_prompt_agent": finally_prompt_agent,
    }

    return lesson


def _build_default_prompt_agent(title: str, content: str, child_name: str) -> str:
    """Build default prompt_agent if not parsed from AI response."""
    return f"""D1: Xin chào {child_name}! Hôm nay Pika sẽ cùng {child_name} học về {title} nhé.
D2: Pika sẽ hướng dẫn {child_name} từng bước một. {child_name} hãy lắng nghe và làm theo nhé.
D3: Bây giờ, {child_name} hãy thử trả lời câu hỏi của Pika nhé!
D4: Pika khen {child_name}: Tuyệt vời! {child_name} học rất nhanh!
D5: Cùng Pika tổng kết: {child_name} đã hoàn thành bài học rồi, giỏi lắm!
→ GOAL: {child_name} hiểu và thực hành được nội dung bài học."""


def _build_finally_prompt_agent(
    child_name: str,
    child_age: Optional[int],
    language: str,
    prompt_agent: str,
    detail_tasks_lesson: str,
) -> str:
    """Build the full finally_prompt_agent matching original API format."""
    # Language mode
    lang_mode = "VIETNAMESE ONLY"
    if language == "en":
        lang_mode = "ENGLISH ONLY"
    elif language == "bi":
        lang_mode = "BILINGUAL"

    # Build activities from detail_tasks_lesson
    activities = detail_tasks_lesson if detail_tasks_lesson else ""

    # Use the template
    finally_prompt = DEFAULT_FINALLY_PROMPT_TEMPLATE.format(
        child_name=child_name,
        child_age=child_age or "",
        language_mode=lang_mode,
        lesson_outline=prompt_agent,
        activities=activities,
    )

    return finally_prompt


def _build_generation_prompt(analysis: dict, custom_prompt: str, language: str) -> str:
    """Build the prompt for lesson generation using improved template."""
    exercise_types = analysis.get("exercise_types", ["short_answer"])
    topic = analysis.get("topic", "General English")
    topic_vi = analysis.get("topic_vietnamese", topic)
    level = analysis.get("level", "Beginner")
    raw_content = analysis.get("raw_content", "")
    suggested_lessons = analysis.get("suggested_lessons", [])
    total_items = analysis.get("total_items", len(suggested_lessons) * 10)

    lang_instruction = {
        "en": "100% tiếng Anh - All content in English only.",
        "vi": "100% tiếng Việt - Toàn bộ nội dung bằng tiếng Việt.",
        "bi": "Song ngữ - Bài tập tiếng Anh, hướng dẫn/giải thích tiếng Việt.",
    }.get(language, "Song ngữ - Bài tập tiếng Anh, hướng dẫn/giải thích tiếng Việt.")

    # Determine lesson count based on content
    lesson_count = min(3, max(1, len(suggested_lessons) or (total_items // 10)))

    prompt = LESSON_GENERATION_PROMPT.format(
        topic=f"{topic} ({topic_vi})",
        exercise_types=', '.join(exercise_types),
        language_instruction=lang_instruction,
        level=level,
        raw_content=raw_content[:3000],
        custom_prompt=custom_prompt or "Tạo bài học vui nhộn, dễ hiểu cho trẻ em",
        suggested_lessons=json.dumps(suggested_lessons, ensure_ascii=False, indent=2) if suggested_lessons else "Tự động chia phù hợp dựa trên nội dung",
        lesson_count=lesson_count,
    )

    return prompt


def _parse_lessons_response(
    response: str,
    request_id: Optional[str],
    image_urls: list[str],
    child_name: str = "Bé",
    language: str = "vi",
) -> list[dict]:
    """Parse lessons from AI response with full format."""
    parsed = _parse_json_from_text(response)

    lessons_data = parsed.get("lessons", [])
    if not lessons_data and isinstance(parsed, list):
        lessons_data = parsed

    lessons = []
    req_id = request_id or str(uuid.uuid4())

    for i, lesson_data in enumerate(lessons_data):
        title = lesson_data.get("title", f"Lesson {i + 1}")
        detail_tasks = lesson_data.get("detail_task_lesson", lesson_data.get("detail_tasks_lesson", ""))

        # Build prompt_agent
        prompt_agent = lesson_data.get("prompt_agent", "")
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
            "summary": lesson_data.get("summary", f"Pika sẽ giúp {child_name} học: {title}"),
            "detail_tasks_lesson": detail_tasks,
            "prompt_agent": prompt_agent,
            "finally_prompt_agent": finally_prompt,
        }
        lessons.append(lesson)

    return lessons


def _parse_json_from_text(text: str) -> dict:
    """Extract and parse JSON from text."""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)

    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
