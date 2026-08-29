"""
Request models for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class ParentConfig(BaseModel):
    """Optional parent configuration."""
    language: Literal["en", "vi", "bi"] = "bi"


class V1GenerateRequest(BaseModel):
    """
    Request for /v1/lessons/generate
    Direct generation from image + custom prompt.
    """
    profile_id: str = Field(..., description="User/child profile ID")
    image_urls: list[str] = Field(..., min_length=1, description="Array of image URLs")
    custom_prompt: str = Field(..., description="User's learning request")
    optional_parent_config: Optional[ParentConfig] = None
    stream: bool = Field(default=True, description="Enable SSE streaming")


class V3GenerateRequest(BaseModel):
    """
    Request for /v3/lessons/generate
    Suggest lessons from images (PDF pages).
    """
    request_id: str = Field(..., description="Unique request ID from BE")
    profile_id: str = Field(..., description="Profile ID")
    profile_api_token: Optional[str] = None
    image_urls: list[str] = Field(..., min_length=1, description="PDF page images")
    language: Optional[Literal["en", "vi", "bi"]] = "bi"  # Top-level language
    learn_agent: Optional[str] = "pika"  # Agent type
    optional_parent_config: Optional[ParentConfig] = None

    def get_language(self) -> str:
        """Get language from either top-level or nested config."""
        if self.language:
            return self.language
        if self.optional_parent_config:
            return self.optional_parent_config.language
        return "bi"


class LessonSuggestion(BaseModel):
    """
    A suggested lesson structure from /v3/generate.
    """
    title: str
    agent_mode: str = "learn_agent"
    template_id: str
    exercise_subtypes: list[str] = []
    content: str
    option: str


class V3GenerateArtifactRequest(BaseModel):
    """
    Request for /v3/lessons/generate_artifact
    Generate full lesson artifacts from suggestions.
    """
    request_id: str
    profile_id: str
    profile_api_token: Optional[str] = None
    stream: bool = True
    lessons: list[LessonSuggestion]
