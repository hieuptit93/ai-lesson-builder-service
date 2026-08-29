from pydantic import BaseModel, Field


class LessonMetadata(BaseModel):
    request_id: str = ""
    processing_time_ms: int = 0
    vision_model: str = "gpt-4o"
    generation_model: str = "gpt-4.1"
    memory_facts_count: int = 0
    content_safety_passed: bool = True
    pipeline_version: str = "2.0.0"


class GenerateLessonResponse(BaseModel):
    request_id: str = ""
    status: str = "success"
    data: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    request_id: str = ""
    status: str = "error"
    error: dict = Field(default_factory=dict)
