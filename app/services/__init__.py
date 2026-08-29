from .ai_client import ai_client, AIClient
from .vision import analyze_images, detect_exercise_subtypes
from .template_matcher import (
    select_template,
    get_option_description,
    get_template_info,
    validate_template_subtype_compatibility,
)
from .lesson_generator import (
    generate_lessons_from_analysis,
    generate_artifacts_from_suggestions,
)
from .sse_streamer import (
    SSEStreamer,
    stream_generation,
    stream_artifact_generation,
)

__all__ = [
    # AI Client
    "ai_client",
    "AIClient",
    # Vision
    "analyze_images",
    "detect_exercise_subtypes",
    # Template Matcher
    "select_template",
    "get_option_description",
    "get_template_info",
    "validate_template_subtype_compatibility",
    # Lesson Generator
    "generate_lessons_from_analysis",
    "generate_artifacts_from_suggestions",
    # SSE Streamer
    "SSEStreamer",
    "stream_generation",
    "stream_artifact_generation",
]
