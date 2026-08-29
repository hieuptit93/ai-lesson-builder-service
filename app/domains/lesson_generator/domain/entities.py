from pydantic import BaseModel, Field
from uuid import uuid4


class AdaptiveRule(BaseModel):
    if_wrong: str = "simplify"
    if_correct: str = "praise_and_continue"
    if_no_response: str = "give_hint"
    max_retries: int = 2


class QuizQuestion(BaseModel):
    question: str = ""
    statement: str = ""
    answer: str | bool = ""
    explanation: str = ""


class Activity(BaseModel):
    phase: str
    type: str
    title: str
    prompt_template: str
    duration_min: int
    questions: list[QuizQuestion] | None = None
    adaptive_rules: AdaptiveRule = Field(default_factory=AdaptiveRule)


class LessonPlan(BaseModel):
    lesson_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    subject: str
    language: str
    difficulty: str = "beginner"
    estimated_duration_min: int = 10
    vocabulary: list[dict] = Field(default_factory=list)
    concepts: list[dict] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    teaching_notes: str | None = None
