"""
Template Matcher Service.

Matches detected exercise types to appropriate lesson templates.
"""
from typing import Optional
from loguru import logger

from app.config import TEMPLATE_REGISTRY, OPTION_DESCRIPTIONS


def select_template(subtypes: list[str], content: str = "") -> str:
    """
    Select the best template_id based on detected subtypes and content.

    Priority order:
    1. Reading Comprehension (if reading_question detected)
    2. Vocabulary (if vocabulary focus)
    3. Sentence Pattern (if sentence structure focus)
    4. Phonics (if pronunciation focus)
    5. Exercise Solver (default - handles most types)
    """
    content_lower = content.lower()

    # Priority 1: Reading Comprehension
    if "reading_question" in subtypes:
        return "ptl_learn_reading_comprehension_v1"

    # Priority 2: Vocabulary
    if "vocabulary_matching" in subtypes or any(
        word in content_lower for word in ["vocabulary", "từ vựng", "word list", "flashcard"]
    ):
        return "ptl_learn_vocab_flashcard_v1"

    # Priority 3: Sentence Pattern
    if "sentence_ordering" in subtypes or any(
        word in content_lower for word in ["pattern", "structure", "mẫu câu", "sentence structure"]
    ):
        return "ptl_learn_sentence_pattern_practice_v1"

    # Priority 4: Phonics
    if "spelling" in subtypes and any(
        word in content_lower for word in ["sound", "phonics", "âm", "pronunciation", "phát âm"]
    ):
        return "ptl_learn_phonics_pronunciation_v1"

    # Check for talk templates
    if any(word in content_lower for word in ["roleplay", "nhập vai", "dialogue", "hội thoại"]):
        return "ptl_talk_roleplay_v1"

    if any(word in content_lower for word in ["presentation", "thuyết trình", "speak", "present"]):
        return "ptl_talk_speaking_presentation_v1"

    if any(word in content_lower for word in ["story", "kể chuyện", "storytelling", "creative"]):
        return "ptl_talk_storytelling_v1"

    # Default: Exercise Solver (handles most exercise types)
    return "ptl_learn_exercise_solver_v1"


def get_option_description(subtypes: list[str]) -> str:
    """
    Get Vietnamese description for the exercise type.
    Used as the 'option' field in lesson suggestions.
    """
    if not subtypes:
        return "Bài tập tổng hợp"

    # Return description for first subtype
    primary_subtype = subtypes[0]
    return OPTION_DESCRIPTIONS.get(primary_subtype, "Bài tập")


def get_template_info(template_id: str) -> Optional[dict]:
    """Get template information by ID."""
    return TEMPLATE_REGISTRY.get(template_id)


def validate_template_subtype_compatibility(template_id: str, subtypes: list[str]) -> bool:
    """
    Check if the subtypes are compatible with the template.
    """
    template = TEMPLATE_REGISTRY.get(template_id)
    if not template:
        return False

    compatible = template.get("compatible_subtypes", [])

    # If template has no restrictions (empty list), allow all
    if not compatible:
        return True

    # Check if any subtype is compatible
    for subtype in subtypes:
        if subtype in compatible:
            return True

    return False
