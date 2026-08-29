"""
Vision Analysis Service.

Analyzes images to extract content and detect exercise types.
"""
import json
import re
from typing import Optional
from loguru import logger

from app.services.ai_client import ai_client
from app.config import EXERCISE_SUBTYPES
from app.prompts import VISION_SYSTEM_PROMPT, VISION_ANALYSIS_PROMPT


async def analyze_images(
    image_urls: list[str],
    custom_prompt: Optional[str] = None,
    language: str = "bi",
) -> dict:
    """
    Analyze images to extract content and structure.

    Args:
        image_urls: List of image URLs to analyze
        custom_prompt: Optional user prompt for context
        language: Target language (en, vi, bi)

    Returns:
        Parsed analysis result as dict
    """
    logger.info(f"Analyzing {len(image_urls)} images, language={language}")

    # Build the prompt
    prompt = VISION_ANALYSIS_PROMPT

    if custom_prompt:
        prompt += f"\n\nYÊU CẦU BỔ SUNG TỪ NGƯỜI DÙNG:\n{custom_prompt}"

    prompt += f"\n\nNGÔN NGỮ MỤC TIÊU: {language}"

    # Call vision AI
    response = await ai_client.analyze_image(
        image_urls=image_urls,
        prompt=prompt,
        system_prompt=VISION_SYSTEM_PROMPT,
    )

    logger.debug(f"Vision response length: {len(response)}")

    # Parse JSON from response
    result = _parse_json_response(response)

    return result


def _parse_json_response(response: str) -> dict:
    """Extract and parse JSON from AI response."""
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
            # Return default structure
            logger.warning("Could not parse JSON from vision response")
            return {
                "exercise_types": ["short_answer"],
                "topic": "General",
                "language": "en",
                "raw_content": response,
                "content_blocks": [],
                "suggested_lessons": [],
            }

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {
            "exercise_types": ["short_answer"],
            "topic": "General",
            "language": "en",
            "raw_content": response,
            "content_blocks": [],
            "suggested_lessons": [],
            "parse_error": str(e),
        }


def detect_exercise_subtypes(content: str) -> list[str]:
    """
    Detect exercise subtypes from content text.
    Uses keyword matching.
    """
    subtypes = []
    content_lower = content.lower()

    # Check each subtype's keywords
    for subtype_id, subtype_info in EXERCISE_SUBTYPES.items():
        keywords = subtype_info.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in content_lower:
                if subtype_id not in subtypes:
                    subtypes.append(subtype_id)
                break

    # Special pattern detection
    # Grammar fill blank: look for blank patterns
    if re.search(r'[.…_]{3,}', content):
        if "grammar_fill_blank" not in subtypes:
            subtypes.append("grammar_fill_blank")

    # Multiple choice: look for option patterns
    if re.search(r'[aA-dD][.)]\s', content) or re.search(r'\([aA-dD]\)', content):
        if "multiple_choice" not in subtypes:
            subtypes.append("multiple_choice")

    # Default to short_answer if nothing detected
    if not subtypes:
        subtypes = ["short_answer"]

    return subtypes
