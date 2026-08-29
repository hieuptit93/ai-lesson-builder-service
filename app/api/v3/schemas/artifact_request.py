from pydantic import BaseModel, Field

from app.core.enums import AgentMode


class LessonItem(BaseModel):
    """A single lesson to be processed into an artifact."""

    title: str = Field(..., min_length=1, description="Lesson title, e.g. 'Lesson 1: Comparative adjectives'")
    agent_mode: AgentMode
    template_id: str = Field(..., min_length=1, description="Template ID to use for this lesson, e.g. 'ptl_learn_exercise_solver_v1'")
    content: str = Field(..., min_length=1, description="Lesson body text, exercises, or instructions")
    option: str = Field(..., min_length=1, description="Selected action option, e.g. 'Solve exercises'")
    exercise_subtypes: list[str] = []


class GenerateArtifactRequest(BaseModel):
    request_id: str | None = None
    profile_id: str
    profile_api_token: str | None = None
    custom_learn_prompt: str | None = None
    custom_talk_prompt: str | None = None
    stream: bool = False
    lessons: list[LessonItem] = Field(..., min_length=1, description="List of lessons to process")
