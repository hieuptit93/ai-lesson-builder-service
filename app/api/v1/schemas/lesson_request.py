from pydantic import BaseModel, Field
from app.core.constants import MAX_IMAGES


class ParentConfig(BaseModel):
    """Optional parent configuration - all fields flexible, any string value accepted."""
    subject: str | None = None  # Any string allowed
    purpose: str | None = None  # Any string allowed
    language: str = "vi"  # Default to Vietnamese
    custom_prompt: str | None = None
    child_age: int | None = Field(default=None, ge=4, le=12)
    child_name: str | None = None


class LessonForRegeneration(BaseModel):
    """Schema for a lesson that needs regeneration."""
    lesson_id: str | None = None
    title: str
    summary: str
    detail_tasks_lesson: str
    prompt_agent: str
    finally_prompt_agent: str | None = None
    finally_title: str | None = None


class GenerateLessonRequest(BaseModel):
    request_id: str | None = None
    profile_id: str  # Required - unified ID for profile
    profile_api_token: str | None = None  # Optional - API token for profile fetching
    image_urls: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)
    custom_prompt: str | None = None  # Optional, top-level
    optional_parent_config: ParentConfig | None = Field(default=None)
    stream: bool = False

    @property
    def parent_config(self) -> ParentConfig:
        """Helper for standard access, ensures we always have a config object."""
        return self.optional_parent_config or ParentConfig()


class RegenerateLessonRequest(GenerateLessonRequest):
    """Inherits all fields from GenerateLessonRequest.
    Adds stream=True default and required lessons_needing_regeneration.
    """
    stream: bool = True
    lessons_needing_regeneration: LessonForRegeneration = Field(
        ..., description="The lesson that needs regeneration"
    )
