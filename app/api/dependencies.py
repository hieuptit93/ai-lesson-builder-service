from functools import lru_cache
import os

from openai import AsyncOpenAI

from app.core.config import Settings
from app.domains.lesson_generator.application.services.generator_service import GeneratorService
from app.domains.lesson_generator.infrastructure.openai_lesson_adapter import OpenAILessonAdapter
from app.domains.memory.application.services.memory_service import MemoryService
from app.domains.memory.infrastructure.mem0_client import Mem0Client
from app.domains.profile.application.services.profile_service import ProfileService
from app.domains.profile.infrastructure.profile_client import ProfileClient
from app.domains.vision_extract.application.services.extraction_service import ExtractionService
from app.domains.vision_extract.infrastructure.openai_vision_adapter import OpenAIVisionAdapter
from app.pipeline.lesson_pipeline import LessonPipeline
from app.pipeline.mock_pipeline import MockPipeline


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache()
def get_pipeline() -> LessonPipeline:
    settings = get_settings()

    # Check for mock mode
    if os.getenv("APP_PIPELINE_TYPE") == "mock":
        return MockPipeline()  # type: ignore

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    vision_adapter = OpenAIVisionAdapter(
        client=openai_client,
        model=settings.openai_vision_model,  # gpt-5.6-terra: quality vision extraction
        guardrail_model=settings.openai_guardrail_model,  # gpt-5.6-luna: ultra cheap guardrail
        temperature=settings.openai_vision_temperature,
        max_tokens=settings.openai_vision_max_tokens,
    )
    extraction_service = ExtractionService(adapter=vision_adapter)

    lesson_adapter = OpenAILessonAdapter(
        client=openai_client,
        model=settings.openai_lesson_model,
        temperature=settings.openai_lesson_temperature,
        max_tokens=settings.openai_lesson_max_tokens,
    )
    generator_service = GeneratorService(adapter=lesson_adapter)

    mem0_client = Mem0Client(base_url=settings.mem0_base_url, timeout=settings.mem0_timeout)
    memory_service = MemoryService(mem0_client=mem0_client)

    profile_client = ProfileClient(
        base_url=settings.profile_api_base_url,
        api_token=settings.profile_api_token,
    )
    profile_service = ProfileService(profile_client=profile_client)

    return LessonPipeline(
        extraction_service=extraction_service,
        generator_service=generator_service,
        memory_service=memory_service,
        profile_service=profile_service,
        settings=settings,
    )
