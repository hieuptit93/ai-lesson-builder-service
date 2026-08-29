from pydantic import BaseModel, Field


class ExtractedContent(BaseModel):
    """Content extracted from images via vision model."""
    raw_text: str = ""
    topic_detected: str = ""
    subject_detected: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SafetyCheckResult(BaseModel):
    """Result of content safety check."""
    is_safe: bool = True
    reason: str = ""
    flagged_categories: list[str] = Field(default_factory=list)
