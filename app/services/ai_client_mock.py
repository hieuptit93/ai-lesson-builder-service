"""
Mock AI Client for testing without real API keys.
"""
import json
import asyncio
from typing import AsyncGenerator, Optional
from loguru import logger


class MockAIClient:
    """Mock AI client that returns realistic test data."""

    def __init__(self):
        self.provider = "mock"
        self.model = "mock-vision"
        logger.info("Using MOCK AI Client (no real API calls)")

    async def analyze_image(
        self,
        image_urls: list[str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Mock image analysis - returns realistic test data."""
        logger.info(f"[MOCK] Analyzing {len(image_urls)} images")

        # Simulate processing time
        await asyncio.sleep(1)

        # Return mock analysis result
        mock_result = {
            "exercise_types": ["grammar_fill_blank", "multiple_choice"],
            "topic": "Pronouns and Be Verbs",
            "language": "en",
            "raw_content": """Fill in the blanks with he, she, it, we, or they:
1. cat and horse………….
2. Mary……………
3. Tom……………
4. Jack and I……………
5. books……………
6. sister……………
7. You and Dave……………
8. plane……………
9. sunshine……………
10. cheese……………""",
            "content_blocks": [
                {
                    "type": "instruction",
                    "text": "Fill in the blanks with he, she, it, we, or they"
                },
                {
                    "type": "exercises",
                    "items": [
                        {"number": 1, "question": "cat and horse………….", "answer_hint": "they"},
                        {"number": 2, "question": "Mary……………", "answer_hint": "she"},
                        {"number": 3, "question": "Tom……………", "answer_hint": "he"},
                        {"number": 4, "question": "Jack and I……………", "answer_hint": "we"},
                        {"number": 5, "question": "books……………", "answer_hint": "they"},
                    ]
                }
            ],
            "suggested_lessons": [
                {
                    "title": "Lesson 1: Fill in the Blanks with Pronouns",
                    "exercise_count": 10,
                    "focus": "Subject pronouns (he, she, it, we, they)"
                },
                {
                    "title": "Lesson 2: Practice with AM, IS, ARE",
                    "exercise_count": 10,
                    "focus": "Be verbs conjugation"
                }
            ]
        }

        return f"```json\n{json.dumps(mock_result, ensure_ascii=False, indent=2)}\n```"

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Mock text generation - returns realistic lesson content."""
        logger.info("[MOCK] Generating text")

        await asyncio.sleep(0.5)

        mock_lessons = {
            "lessons": [
                {
                    "title": "Lesson 1: Fill in the Blanks with Pronouns",
                    "summary": "Học cách sử dụng đại từ nhân xưng (he, she, it, we, they) để thay thế danh từ trong câu.",
                    "detail_task_lesson": """Trong bài học này, con sẽ học cách dùng đại từ nhân xưng:

- HE: dùng cho nam (boy, man, father, Tom...)
- SHE: dùng cho nữ (girl, woman, mother, Mary...)
- IT: dùng cho vật, động vật (cat, book, plane...)
- WE: dùng cho "tôi và bạn/họ" (Jack and I, you and I...)
- THEY: dùng cho nhiều người/vật (cats, books, Mary and Tom...)

Hãy điền đại từ phù hợp vào chỗ trống!""",
                    "exercise_subtypes": ["grammar_fill_blank"]
                },
                {
                    "title": "Lesson 2: Practice with AM, IS, ARE",
                    "summary": "Luyện tập cách chia động từ TO BE (am, is, are) với các chủ ngữ khác nhau.",
                    "detail_task_lesson": """Quy tắc chia động từ TO BE:

- I + AM
- He/She/It + IS
- We/You/They + ARE

Ví dụ:
- I am a student.
- She is beautiful.
- They are friends.

Hãy điền am, is, are vào chỗ trống!""",
                    "exercise_subtypes": ["grammar_fill_blank"]
                }
            ]
        }

        return f"```json\n{json.dumps(mock_lessons, ensure_ascii=False, indent=2)}\n```"

    async def generate_text_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Mock streaming text generation."""
        logger.info("[MOCK] Streaming text")

        text = await self.generate_text(prompt, system_prompt)

        # Stream character by character with small delay
        for char in text:
            yield char
            await asyncio.sleep(0.01)


# For easy switching
mock_ai_client = MockAIClient()
