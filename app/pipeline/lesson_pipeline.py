"""LessonPipeline: orchestrates Memory -> Vision -> Lesson Generation.

Supports both sync (JSON) and SSE streaming modes.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import structlog

from app.api.v3.schemas.artifact_request import GenerateArtifactRequest
from app.api.v1.schemas.lesson_request import GenerateLessonRequest, RegenerateLessonRequest
from app.core.config import Settings
from app.core.enums import AgentBotId
from app.domains.lesson_generator.application.services.generator_service import GeneratorService
from app.domains.lesson_generator.application.services.prompt_builder import (
    CONTEXT_STYLE_GUIDELINE_PROMPT_TEMPLATE,
    USER_PROFILE_PROMPT_TEMPLATE,
    get_langfuse_prompt,
    _get_language_prompt,
    _get_task_base_on_language_prompt,
)
from app.domains.memory.application.services.memory_service import MemoryService
from app.domains.profile.application.services.profile_service import ProfileService
from app.core.exceptions import AIServiceError, MissingImageUrlsError, UnsafeContentError, VisionExtractionError
from app.domains.vision_extract.application.services.extraction_service import ExtractionService
from app.utils.cost import estimate_cost

logger = structlog.get_logger()

_DEFAULT_FEEDBACK_POLICY = {
    "correct": "That's correct! Great job!",
    "incorrect": "That's not quite right. Would you like to try a different answer?",
    "no_response": "Keep going! Would you like to take a guess?",
}


def _enrich_card_specs(card_specs: list, exercise_subtypes: list[str] | None = None) -> list:
    primary = (exercise_subtypes or [None])[0]
    return [
        {"visual_placeholder": "", "audio_placeholder": "", "exercise_subtype": primary, "feedback_policy": _DEFAULT_FEEDBACK_POLICY, **card}
        for card in card_specs
    ]


_CHECKPOINT_STRIP_FIELDS = {"accepted_answers", "explanation"}


def _enrich_checkpoint_specs(checkpoint_specs: list, exercise_subtypes: list[str] | None = None) -> list:
    primary = (exercise_subtypes or [None])[0]
    return [
        checkpoint if checkpoint.get("type") == "narrative"
        else {"exercise_subtype": primary, **{k: v for k, v in checkpoint.items() if k not in _CHECKPOINT_STRIP_FIELDS}}
        for checkpoint in checkpoint_specs
    ]


template_option_labels = {
    "ptl_learn_vocab_flashcard_v1": "Vocabulary",
    "ptl_learn_phonics_pronunciation_v1": "Pronunciation",
    "ptl_learn_exercise_solver_v1": "Solve exercises",
    "ptl_learn_sentence_pattern_practice_v1": "Sentence patterns",
    "ptl_learn_reading_comprehension_v1": "Reading comprehension",
    "ptl_talk_roleplay_v1": "Roleplay conversation",
    "ptl_talk_speaking_presentation_v1": "Presentation",
    "ptl_talk_storytelling_v1": "Creative storytelling",
}


def _enrich_options(options: list) -> list:
    return [
        {**opt, "option": template_option_labels.get(opt.get("template_id", ""), "")}
        for opt in options
    ]


def _transform_lesson_plan(
    lesson_plan: dict,
    child_name: str | None = None,
    child_age: int | None = None,
    language: str | None = None,
    favorite_movie: str | None = None,
    dynamic_memory: str | None = None,
) -> dict:
    """Transform lesson plan: add IDs and create finally_prompt_agent."""
    if "lessons" not in lesson_plan:
        return lesson_plan

    # Try to fetch prompts from Langfuse, fallback to local constants
    context_style = get_langfuse_prompt("context_style_guideline_prompt")
    if context_style is None:
        context_style = CONTEXT_STYLE_GUIDELINE_PROMPT_TEMPLATE

    user_profile_template = get_langfuse_prompt("user_profile_prompt")
    if user_profile_template is None:
        user_profile_template = USER_PROFILE_PROMPT_TEMPLATE

    # Resolve child name ONCE - same value everywhere in the prompt.
    # "bạn nhỏ" is a natural Vietnamese fallback; "friend"/"N/A" mixed usage
    # previously made the agent see two conflicting names.
    resolved_name = child_name or "bạn nhỏ"

    # Format user profile with actual values
    user_profile_str = user_profile_template.format(
        child_name=resolved_name,
        child_age=child_age or "N/A",
        favorite_movie=favorite_movie or "N/A",
    )

    # Map language code to language mode for prompt
    language_mode_map = {
        "vi": "Vietnamese",
        "en": "English",
        "bi": "Vietnamese and English (bilingual)",
        "bilingual": "Vietnamese and English (bilingual)",
    }
    normalized_language = "bi" if language == "bilingual" else (language or "vi")
    language_mode = language_mode_map.get(language, "Vietnamese")

    # Get language-specific prompt for Pika response
    language_prompt = _get_language_prompt(normalized_language)

    # Get language-specific task instruction for detail_tasks_lesson
    tasks_for_language = _get_task_base_on_language_prompt(normalized_language)

    def _fill_placeholders(text: str) -> str:
        """Replace {{...}} placeholders. Applied to BOTH context template and
        language prompt - the bilingual language prompt contains {{name}} too,
        which previously leaked unreplaced into finally_prompt_agent."""
        return (
            text.replace("{{name}}", resolved_name)
            .replace("{{age}}", str(child_age or ""))
            .replace("{{favorite_movie}}", favorite_movie or "")
            .replace("{{LANGUAGE_MODE}}", language_mode)
            .replace("{{dynamic_memory}}", dynamic_memory or "none")
        )

    # Pre-compute filled prompts once (same for all lessons)
    context_filled = _fill_placeholders(context_style)
    language_prompt = _fill_placeholders(language_prompt)

    for idx, lesson in enumerate(lesson_plan["lessons"], start=1):
        # Add lesson_id if missing
        lesson.setdefault("lesson_id", f"lesson_{idx:03d}")

        # Create finally_prompt_agent from prompt_agent
        prompt_agent = lesson.get("prompt_agent", "")
        if prompt_agent:
            detail_tasks = lesson.get("detail_tasks_lesson", "")
            if tasks_for_language:
                detail_tasks = f"{detail_tasks}\n\n{tasks_for_language}" if detail_tasks else tasks_for_language

            lesson["detail_tasks_lesson"] = detail_tasks

            lesson["finally_prompt_agent"] = (
                f"{context_filled}\n\n{language_prompt}\n\n{prompt_agent}\n\n{detail_tasks}\n\n{user_profile_str}"
            )

    return lesson_plan


def _build_personalization_context(
    child_name: str | None,
    child_age: int | None,
    learning_history: list[str] | None,
    memory_facts: list | None,
) -> str | None:
    """Build the CHILD PERSONALIZATION section injected into the vision prompt.

    Mirrors the source system's MEMORY_SECTION/PARENT_SECTION: profile + mem0
    facts feed the GENERATION step so the produced lesson content itself is
    personalized (difficulty by age, examples from interests, review links to
    recently learned vocabulary).
    """
    parts: list[str] = []
    if child_name:
        parts.append(f"- Child name: {child_name} — use the name naturally in summary and detail_tasks_lesson instead of generic \"bé\"")
    if child_age:
        parts.append(f"- Child age: {child_age} — adjust difficulty, sentence length, and tone for this age")
    if learning_history:
        vocab = ", ".join(learning_history[:20])
        parts.append(f"- Recently learned vocabulary: {vocab} — do NOT re-teach these as new words; you may reference them for review or to connect with new content")
    if memory_facts:
        facts_text = "\n".join(f"  - {f.text}" for f in memory_facts[:10])
        parts.append(f"- Facts remembered about the child (from memory):\n{facts_text}\n  Use these to personalize examples and encouragement where natural")

    if not parts:
        return None

    return (
        "## CHILD PERSONALIZATION (apply when generating summary, detail_tasks_lesson, prompt_agent)\n"
        + "\n".join(parts)
        + "\nPersonalize ONLY style, examples, difficulty and encouragement. "
        "NEVER change, add, or invent lesson content beyond what is visible in the images."
    )


def _enforce_single_lesson(lesson_plan: dict) -> dict:
    """Server-side enforcement: ensure exactly 1 lesson with lesson_id='lesson_regen'."""
    lessons = lesson_plan.get("lessons", [])
    if len(lessons) > 1:
        logger.warning(
            "pipeline.regenerate.lessons_clipped",
            original_count=len(lessons),
            kept=1,
        )
        lesson_plan["lessons"] = [lessons[0]]

    if lesson_plan.get("lessons"):
        lesson_plan["lessons"][0]["lesson_id"] = "lesson_regen"

    return lesson_plan


def _sse(payload: dict, event: str | None = None) -> str:
    """Build SSE message with optional event field."""
    if event:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class LessonPipeline:
    # Expert avatar URLs mapping
    EXPERT_AVATARS = {
        "vision_analyst": "expert_avatar_vision_analyst",
        "curriculum_designer": "expert_avatar_curriculum_designer",
        "child_psychologist": "expert_avatar_child_psychologist",
        "safety_reviewer": "expert_avatar_safety_reviewer",
        "final_editor": "expert_avatar_final_editor",
    }

    def __init__(
        self,
        extraction_service: ExtractionService,
        generator_service: GeneratorService,
        memory_service: MemoryService,
        profile_service: ProfileService,
        settings: Settings | None = None,
    ):
        self._extraction = extraction_service
        self._generator = generator_service
        self._memory = memory_service
        self._profile = profile_service
        self._settings = settings

    # ------------------------------------------------------------------
    # Sync mode
    # ------------------------------------------------------------------
    async def generate(self, request: GenerateLessonRequest, *, request_id: str) -> dict:
        start = time.monotonic()

        logger.info(
            "pipeline.start",
            log_type="job",
            feature="LESSON",
            request_id=request_id,
            profile_id=request.profile_id,
            has_images=bool(request.image_urls),
            subject=request.optional_parent_config.subject if request.optional_parent_config else "english",
        )

        parent_config = request.optional_parent_config
        subject = parent_config.subject if parent_config else "english"
        purpose = parent_config.purpose if parent_config else "review"
        custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)

        # Run all 3 APIs in parallel: Profile, Memory, Vision
        async def fetch_profile():
            return await self._profile.fetch_profile(
                profile_id=request.profile_id,
                token=request.profile_api_token if hasattr(request, 'profile_api_token') else None,
            )

        async def fetch_memory():
            return await self._memory.fetch_user_memory(
                user_id=request.profile_id,
                topic=custom_prompt or subject,
                subject=subject,
            )

        async def extract_vision():
            if request.image_urls:
                return await self._extraction.extract_from_images(
                    image_urls=request.image_urls,
                    subject_hint=subject,
                )
            return ""

        user_profile, memory, image_description = await asyncio.gather(
            fetch_profile(),
            fetch_memory(),
            extract_vision(),
        )

        is_mock_data = user_profile.is_degraded

        if user_profile.child:
            child_age = user_profile.child.age or (parent_config.child_age if parent_config else None)
            child_name = user_profile.child.child_name or (parent_config.child_name if parent_config else None)
        else:
            child_age = parent_config.child_age if parent_config else None
            child_name = parent_config.child_name if parent_config else None

        language = (parent_config.language if parent_config and parent_config.language else None) or user_profile.language_preference or "vi"

        extracted_content = {
            "raw_text": image_description or custom_prompt or "",
            "topic_detected": custom_prompt or subject,
            "subject_detected": subject,
        }

        expert_log, lesson_plan = await self._generator.generate_lesson(
            extracted_content=extracted_content,
            subject=subject,
            purpose=purpose,
            language=language,
            memory_facts=memory.facts if memory.facts else None,
            parent_notes=custom_prompt,
            child_age=child_age,
            child_name=child_name,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None

        lesson_plan = _transform_lesson_plan(
            lesson_plan,
            child_name=child_name,
            child_age=child_age,
            language=language,
            favorite_movie=None,
            dynamic_memory=dynamic_memory,
        )

        data = {
            **lesson_plan,
            "metadata": {
                "request_id": request_id,
                "profile_id": request.profile_id,
                "child_name": child_name,
                "child_age": child_age,
                "language": language,
                "model_used": "gpt-4o + gpt-4.1",
                "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                "processing_time_ms": elapsed_ms,
                "content_safety_passed": True,
                "pipeline_version": "2.0.0",
                "is_mock_data": is_mock_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expert_discussion_log": expert_log,
                "vision_extracted_text": image_description,
            },
        }

        logger.info(
            "pipeline.success",
            log_type="job",
            feature="LESSON",
            request_id=request_id,
            profile_id=request.profile_id,
            duration_ms=elapsed_ms,
            lessons_count=len(lesson_plan.get("lessons", [])),
            is_mock_data=is_mock_data,
        )

        return {"request_id": request_id, "status": "success", "data": data}

    async def generate_v3(self, request: GenerateLessonRequest, *, request_id: str) -> dict:
        start = time.monotonic()

        logger.info(
            "pipeline.start",
            log_type="job",
            feature="LESSON",
            request_id=request_id,
            profile_id=request.profile_id,
            has_images=bool(request.image_urls),
            subject=request.optional_parent_config.subject if request.optional_parent_config else "english",
        )

        parent_config = request.optional_parent_config
        subject = parent_config.subject if parent_config else "english"
        purpose = parent_config.purpose if parent_config else "review"
        custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)
        language = (parent_config.language if parent_config and parent_config.language else None) or "vi"

        try:
            async def fetch_profile():
                return await self._profile.fetch_profile(
                    profile_id=request.profile_id,
                    token=request.profile_api_token if hasattr(request, 'profile_api_token') else None,
                )

            if not request.image_urls:
                raise MissingImageUrlsError()

            # Guardrail: Skip if disabled in settings (for trusted sources)
            guardrail_enabled = self._settings.openai_guardrail_enabled if self._settings else True

            async def run_guardrail() -> str:
                """Returns rejection reason, or empty string when safe/disabled."""
                if not guardrail_enabled:
                    logger.info("guardrail_skipped", request_id=request_id, reason="disabled_in_settings")
                    return ""
                try:
                    await self._extraction.check_images_safety_v3(request.image_urls)
                    return ""
                except UnsafeContentError as exc:
                    return exc.message

            async def fetch_memory():
                return await self._memory.fetch_user_memory(
                    user_id=request.profile_id,
                    topic=custom_prompt or subject,
                    subject=subject,
                )

            # Phase 1: guardrail + profile + memory in PARALLEL (all fast/cheap).
            # Profile & memory must complete BEFORE vision so the generation
            # step receives child personalization context (source-system parity).
            guardrail_detail, user_profile, memory = await asyncio.gather(
                run_guardrail(),
                fetch_profile(),
                fetch_memory(),
            )

            if guardrail_detail:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "data": {
                        "rejected": True,
                        "reason_code": "unsafe_content",
                        "reason": guardrail_detail,
                        "content": "",
                        "lessons": [],
                        "metadata": {
                            "language": language,
                            "processing_time_ms": elapsed_ms,
                            "content_safety_passed": False,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "token_usage": None,
                            "cost_usd": 0.0,
                        },
                    },
                }

            # Resolve child info BEFORE vision call so it can personalize generation
            if user_profile.child:
                child_age = user_profile.child.age or (parent_config.child_age if parent_config else None)
                child_name = user_profile.child.child_name or (parent_config.child_name if parent_config else None)
            else:
                child_age = parent_config.child_age if parent_config else None
                child_name = parent_config.child_name if parent_config else None
            learning_history = user_profile.child.learning_history if user_profile.child else []

            personalization_context = _build_personalization_context(
                child_name=child_name,
                child_age=child_age,
                learning_history=learning_history,
                memory_facts=memory.facts,
            )

            # Phase 2: vision + lesson generation in ONE call, now personalized
            extracted_result, token_usage = await self._extraction.extract_from_images_v3(
                image_urls=request.image_urls,
                subject_hint=subject,
                custom_prompt=custom_prompt,
                personalization_context=personalization_context,
            )

            # Validate and normalize response format
            if not extracted_result:
                raise VisionExtractionError(
                    "Vision model returned empty response",
                    raw_response=None,
                )

            # Check if content was rejected by guardrail in vision model
            if extracted_result.get("rejected"):
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "data": {
                        "rejected": True,
                        "reason_code": extracted_result.get("reason_code", "content_rejected"),
                        "reason": extracted_result.get("reason", "Content was rejected"),
                        "content": extracted_result.get("content", ""),
                        "lessons": [],
                        "metadata": {
                            "language": language,
                            "processing_time_ms": elapsed_ms,
                            "content_safety_passed": False,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "token_usage": token_usage,
                            "cost_usd": 0.0,
                        },
                    },
                }

            if "lessons" not in extracted_result:
                # Log the actual keys for debugging
                actual_keys = list(extracted_result.keys()) if isinstance(extracted_result, dict) else "not_dict"
                logger.warning(
                    "vision_response_wrong_format",
                    expected_key="lessons",
                    actual_keys=actual_keys,
                    response_preview=str(extracted_result)[:300],
                )
                # Try to handle legacy format
                if isinstance(extracted_result, dict) and "suggested_lessons" in extracted_result:
                    extracted_result["lessons"] = extracted_result.pop("suggested_lessons")
                else:
                    raise VisionExtractionError(
                        f"Vision model returned wrong format. Expected 'lessons', got keys: {actual_keys}",
                        raw_response=str(extracted_result)[:500],
                    )

            if not extracted_result.get("lessons"):
                # Model found no teachable content but did not reject - treat as rejection
                logger.warning("vision_v3_empty_lessons", request_id=request_id)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "data": {
                        "rejected": True,
                        "reason_code": "no_educational_content",
                        "reason": "Không tìm thấy nội dung học tập rõ ràng trong ảnh",
                        "content": "",
                        "lessons": [],
                        "metadata": {
                            "language": language,
                            "processing_time_ms": elapsed_ms,
                            "content_safety_passed": True,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "token_usage": token_usage,
                            "cost_usd": 0.0,
                        },
                    },
                }

            for lesson in extracted_result.get("lessons", []):
                lesson["options"] = _enrich_options(lesson.get("options", []))

            is_mock_data = user_profile.is_degraded
            language = (parent_config.language if parent_config and parent_config.language else None) or user_profile.language_preference or "vi"

            # Transform lessons: add lesson_id and finally_prompt_agent
            dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None
            transformed_result = _transform_lesson_plan(
                {"lessons": extracted_result.get("lessons", [])},
                child_name=child_name,
                child_age=child_age,
                language=language,
                favorite_movie=None,
                dynamic_memory=dynamic_memory,
            )

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Get vision model from settings for accurate cost estimation
            vision_model = self._settings.openai_vision_model if self._settings else "gpt-5.6-terra"

            data = {
                "rejected": extracted_result.get("rejected", False),
                "reason_code": extracted_result.get("reason_code"),
                "reason": extracted_result.get("reason", ""),
                "content": extracted_result.get("content", "Đã tạo bài học thành công"),
                "lessons": transformed_result.get("lessons", []),
                "metadata": {
                    "language": language,
                    "child_name": child_name,
                    "child_age": child_age,
                    "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                    "processing_time_ms": elapsed_ms,
                    "content_safety_passed": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_used": vision_model,  # Option B: only vision model, no separate lesson gen
                    "token_usage": token_usage,
                    "cost_usd": estimate_cost(
                        model=vision_model,
                        prompt_tokens=(token_usage or {}).get("prompt_tokens", 0),
                        completion_tokens=(token_usage or {}).get("completion_tokens", 0),
                    ) if token_usage else 0.0,
                },
            }

            logger.info(
                "pipeline.success",
                log_type="job",
                feature="LESSON",
                request_id=request_id,
                profile_id=request.profile_id,
                duration_ms=elapsed_ms,
                is_mock_data=is_mock_data,
            )

            return {"request_id": request_id, "status": "success", "data": data}

        except AIServiceError as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "pipeline.generate_v3.error",
                request_id=request_id,
                error_code=exc.error_code,
                error=exc.message,
            )
            return {
                "request_id": request_id,
                "status": "failed",
                "data": {
                    "rejected": True,
                    "reason_code": exc.error_code,
                    "reason": exc.message,
                    "content": "",
                    "lessons": [],
                    "metadata": {
                        "language": language,
                        "processing_time_ms": elapsed_ms,
                        "content_safety_passed": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "token_usage": None,
                        "cost_usd": 0.0,
                    },
                },
            }
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception(
                "pipeline.generate_v3.error",
                request_id=request_id,
                error=str(exc),
            )
            return {
                "request_id": request_id,
                "status": "failed",
                "data": {
                    "rejected": True,
                    "reason_code": "internal_error",
                    "reason": str(exc),
                    "content": "",
                    "lessons": [],
                    "metadata": {
                        "language": language,
                        "processing_time_ms": elapsed_ms,
                        "content_safety_passed": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "token_usage": None,
                        "cost_usd": 0.0,
                    },
                },
            }

    async def stream_generate_v3(
        self,
        request: GenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        """Stream v3 lesson generation (vision -> suggested_lessons) with SSE."""
        start = time.monotonic()

        try:
            parent_config = request.optional_parent_config
            subject = parent_config.subject if parent_config else "english"
            custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)

            pipeline_image = getattr(self._settings, "pipeline_robot_image_url", "") if self._settings else ""
            yield _sse({"phase": "started", "request_id": request_id, "message": "Pika is analyzing request...", "image_url": pipeline_image}, "pipeline")

            async def fetch_profile():
                return await self._profile.fetch_profile(
                    profile_id=request.profile_id,
                    token=request.profile_api_token if hasattr(request, "profile_api_token") else None,
                )

            if not request.image_urls:
                raise MissingImageUrlsError()

            # Guardrail: Skip if disabled in settings (for trusted sources)
            guardrail_enabled = self._settings.openai_guardrail_enabled if self._settings else True

            async def run_guardrail() -> str:
                """Returns rejection reason, or empty string when safe/disabled."""
                if not guardrail_enabled:
                    logger.info("guardrail_skipped", request_id=request_id, reason="disabled_in_settings")
                    return ""
                try:
                    await self._extraction.check_images_safety_v3(request.image_urls)
                    return ""
                except UnsafeContentError as exc:
                    return exc.message

            async def fetch_memory():
                return await self._memory.fetch_user_memory(
                    user_id=request.profile_id,
                    topic=custom_prompt or subject,
                    subject=subject,
                )

            # Phase 1: guardrail + profile + memory in PARALLEL,
            # so the vision call can be personalized with child context.
            guardrail_detail, user_profile, memory = await asyncio.gather(
                run_guardrail(),
                fetch_profile(),
                fetch_memory(),
            )

            if guardrail_detail:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                language = (parent_config.language if parent_config and parent_config.language else None) or "vi"
                data = {
                    "rejected": True,
                    "reason_code": "unsafe_content",
                    "reason": guardrail_detail,
                    "content": "",
                    "lessons": [],
                    "metadata": {
                        "language": language,
                        "processing_time_ms": elapsed_ms,
                        "content_safety_passed": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "usage": None,
                        "cost_usd": 0.0,
                    },
                }
                yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
                yield "data: [DONE]\n\n"
                return

            # Resolve child info BEFORE vision call so it can personalize generation
            if user_profile.child:
                child_age = user_profile.child.age or (parent_config.child_age if parent_config else None)
                child_name = user_profile.child.child_name or (parent_config.child_name if parent_config else None)
            else:
                child_age = parent_config.child_age if parent_config else None
                child_name = parent_config.child_name if parent_config else None
            learning_history = user_profile.child.learning_history if user_profile.child else []

            yield _sse({
                "phase": "profile",
                "message": f"Profile fetched: {child_name or 'N/A'}",
                "child_name": child_name,
                "child_age": child_age,
                "language": user_profile.language_preference,
                "image_url": pipeline_image,
            }, "pipeline")

            personalization_context = _build_personalization_context(
                child_name=child_name,
                child_age=child_age,
                learning_history=learning_history,
                memory_facts=memory.facts,
            )

            # Phase 2: vision + lesson generation in ONE call, now personalized
            extracted_result, token_usage = await self._extraction.extract_from_images_v3(
                image_urls=request.image_urls,
                subject_hint=subject,
                custom_prompt=custom_prompt,
                personalization_context=personalization_context,
            )

            # Validate and normalize response format
            if not extracted_result:
                raise VisionExtractionError(
                    "Vision model returned empty response",
                    raw_response=None,
                )

            # Check if content was rejected by guardrail in vision model
            if extracted_result.get("rejected"):
                elapsed_ms = int((time.monotonic() - start) * 1000)
                language = (parent_config.language if parent_config and parent_config.language else None) or "vi"
                data = {
                    "rejected": True,
                    "reason_code": extracted_result.get("reason_code", "content_rejected"),
                    "reason": extracted_result.get("reason", "Content was rejected"),
                    "content": extracted_result.get("content", ""),
                    "lessons": [],
                    "metadata": {
                        "language": language,
                        "processing_time_ms": elapsed_ms,
                        "content_safety_passed": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "usage": token_usage,
                        "cost_usd": 0.0,
                    },
                }
                yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
                yield "data: [DONE]\n\n"
                return

            if "lessons" not in extracted_result:
                actual_keys = list(extracted_result.keys()) if isinstance(extracted_result, dict) else "not_dict"
                logger.warning(
                    "vision_response_wrong_format",
                    expected_key="lessons",
                    actual_keys=actual_keys,
                    response_preview=str(extracted_result)[:300],
                )
                # Try to handle legacy format
                if isinstance(extracted_result, dict) and "suggested_lessons" in extracted_result:
                    extracted_result["lessons"] = extracted_result.pop("suggested_lessons")
                else:
                    raise VisionExtractionError(
                        f"Vision model returned wrong format. Expected 'lessons', got keys: {actual_keys}",
                        raw_response=str(extracted_result)[:500],
                    )

            if not extracted_result.get("lessons"):
                # Model found no teachable content but did not reject - treat as rejection
                logger.warning("vision_v3_empty_lessons", request_id=request_id)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                language = (parent_config.language if parent_config and parent_config.language else None) or "vi"
                data = {
                    "rejected": True,
                    "reason_code": "no_educational_content",
                    "reason": "Không tìm thấy nội dung học tập rõ ràng trong ảnh",
                    "content": "",
                    "lessons": [],
                    "metadata": {
                        "language": language,
                        "processing_time_ms": elapsed_ms,
                        "content_safety_passed": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "usage": token_usage,
                        "cost_usd": 0.0,
                    },
                }
                yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
                yield "data: [DONE]\n\n"
                return

            for lesson in extracted_result["lessons"]:
                lesson["options"] = _enrich_options(lesson.get("options", []))

            is_mock_data = user_profile.is_degraded
            language = (
                (parent_config.language if parent_config and parent_config.language else None)
                or user_profile.language_preference
                or "vi"
            )

            if request.image_urls:
                yield _sse({
                    "phase": "vision_complete",
                    "message": "Image analysis complete",
                    "image_url": pipeline_image,
                }, "pipeline")

            # Transform lessons: add lesson_id and finally_prompt_agent
            dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None
            transformed_result = _transform_lesson_plan(
                {"lessons": extracted_result.get("lessons", [])},
                child_name=child_name,
                child_age=child_age,
                language=language,
                favorite_movie=None,
                dynamic_memory=dynamic_memory,
            )

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Get vision model from settings for accurate cost estimation
            vision_model = self._settings.openai_vision_model if self._settings else "gpt-5.6-terra"

            data = {
                "rejected": extracted_result.get("rejected", False),
                "reason_code": extracted_result.get("reason_code"),
                "reason": extracted_result.get("reason", ""),
                "content": extracted_result.get("content", "Đã tạo bài học thành công"),
                "lessons": transformed_result.get("lessons", []),
                "metadata": {
                    "language": language,
                    "child_name": child_name,
                    "child_age": child_age,
                    "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                    "processing_time_ms": elapsed_ms,
                    "content_safety_passed": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_used": vision_model,  # Option B: only vision model, no separate lesson gen
                    "usage": token_usage,
                    "cost_usd": estimate_cost(
                        model=vision_model,
                        prompt_tokens=(token_usage or {}).get("prompt_tokens", 0),
                        completion_tokens=(token_usage or {}).get("completion_tokens", 0),
                    ) if token_usage else 0.0,
                },
            }

            yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            logger.warning("pipeline.cancelled", log_type="job", feature="LESSON", request_id=request_id, profile_id=request.profile_id)
            raise
        except (MissingImageUrlsError, VisionExtractionError) as exc:
            logger.warning("pipeline.vision_error", log_type="job", feature="LESSON", request_id=request_id, error_code=exc.error_code, error=exc.message)
            yield _sse({"phase": "error", "code": exc.error_code, "message": exc.message}, "error")
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("pipeline.error", log_type="job", feature="LESSON", request_id=request_id, profile_id=request.profile_id, error=str(exc))
            yield _sse({"phase": "error", "code": "internal_error", "message": str(exc)}, "error")
            yield "data: [DONE]\n\n"

    async def generate_artifact_v3(self, request: GenerateArtifactRequest, *, request_id: str) -> dict:
        start = time.monotonic()

        logger.info(
            "pipeline.artifact.start",
            log_type="job",
            feature="ARTIFACT",
            request_id=request_id,
            profile_id=request.profile_id,
            lesson_count=len(request.lessons),
        )

        async def fetch_profile():
            return await self._profile.fetch_profile(
                profile_id=request.profile_id,
                token=request.profile_api_token,
            )

        async def fetch_memory():
            return await self._memory.fetch_user_memory(
                user_id=request.profile_id,
                topic=request.custom_learn_prompt or request.custom_talk_prompt or "lesson",
                subject="english",
            )

        user_profile, memory = await asyncio.gather(fetch_profile(), fetch_memory())

        is_mock_data = user_profile.is_degraded
        if user_profile.child:
            child_age = user_profile.child.age
            child_name = user_profile.child.child_name
        else:
            child_age = None
            child_name = None
        language = user_profile.language_preference or "vi"

        artifact_lessons = []
        total_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for idx, lesson_item in enumerate(request.lessons, start=1):
            if lesson_item.agent_mode == "learn_agent":
                artifact_data, lesson_usage = await self._generator.generate_artifact_for_lesson(
                    lesson_title=lesson_item.title,
                    lesson_content=lesson_item.content,
                    lesson_option=lesson_item.option,
                    agent_mode=lesson_item.agent_mode,
                    language=language,
                    child_name=child_name,
                    child_age=child_age,
                    custom_prompt=request.custom_learn_prompt,
                    template_id=lesson_item.template_id,
                )
            else:  # talk_agent
                artifact_data, lesson_usage = await self._generator.generate_talk_agent_system_task(
                    lesson_content=lesson_item.content,
                    template_id=lesson_item.template_id,
                    language=language,
                    child_name=child_name,
                    child_age=child_age,
                    memory_facts=memory.facts if memory.facts else None,
                    custom_prompt=request.custom_talk_prompt,
                )

            lesson_id = f"lesson_{idx:03d}"
            bot_id = AgentBotId[lesson_item.agent_mode.upper()]
            template_id = lesson_item.template_id

            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_token_usage[key] += lesson_usage.get(key, 0)

            artifact_lessons.append({
                "lesson_id": lesson_id,
                "bot_id": bot_id,
                "title": lesson_item.title,
                "summary": artifact_data.get("summary", ""),
                "detail_tasks_lesson": artifact_data.get("detail_tasks_lesson", ""),
                "lesson_json": {
                    "lesson_title": lesson_item.title,
                    "agent_mode": lesson_item.agent_mode,
                    "template_id": template_id,
                    "runtime_bot_template": "PTL_LEARN_GENERIC_BOT_ID" if lesson_item.agent_mode == "learn_agent" else "PTL_TALK_GENERIC_BOT_ID",
                    "system_task_description": artifact_data.get("system_task_description", ""),
                    "card_specs": _enrich_card_specs(artifact_data.get("card_specs", []), lesson_item.exercise_subtypes) if lesson_item.agent_mode == "learn_agent" else artifact_data.get("card_specs", []),
                    "audio_specs": artifact_data.get("audio_specs", []),
                    "checkpoint_specs": _enrich_checkpoint_specs(artifact_data.get("checkpoint_specs", []), lesson_item.exercise_subtypes) if lesson_item.agent_mode == "learn_agent" else artifact_data.get("checkpoint_specs", []),
                },
            })

        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = {
            "rejected": False,
            "reason_code": None,
            "reason": "",
            "content": "Created lessons successfully",
            "lessons": artifact_lessons,
            "metadata": {
                "request_id": request_id,
                "profile_id": request.profile_id,
                "child_name": child_name,
                "child_age": child_age,
                "language": language,
                "model_used": "gpt-4o + gpt-4.1",
                "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                "processing_time_ms": elapsed_ms,
                "content_safety_passed": True,
                "pipeline_version": "2.0.0",
                "is_mock_data": is_mock_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expert_discussion_log": "",
                "vision_extracted_text": None,
                "token_usage": total_token_usage,
                "cost_usd": estimate_cost(
                    model="gpt-4.1",
                    prompt_tokens=total_token_usage["prompt_tokens"],
                    completion_tokens=total_token_usage["completion_tokens"],
                ),
            },
        }

        logger.info(
            "pipeline.artifact.success",
            log_type="job",
            feature="ARTIFACT",
            request_id=request_id,
            profile_id=request.profile_id,
            duration_ms=elapsed_ms,
            lessons_count=len(artifact_lessons),
            is_mock_data=is_mock_data,
        )

        return {"request_id": request_id, "status": "success", "data": data}

    async def stream_artifact_v3(
        self,
        request: GenerateArtifactRequest,
        *,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """Stream artifact generation with SSE, yielding each lesson as it completes."""
        start = time.monotonic()
        total_lessons = len(request.lessons)

        try:
            pipeline_image = getattr(self._settings, "pipeline_robot_image_url", "") if self._settings else ""
            yield _sse({
                "phase": "started",
                "request_id": request_id,
                "message": f"Pika is creating artifacts for {total_lessons} lessons...",
                "total_lessons": total_lessons,
                "image_url": pipeline_image,
            }, "pipeline")

            async def fetch_profile():
                return await self._profile.fetch_profile(
                    profile_id=request.profile_id,
                    token=request.profile_api_token,
                )

            async def fetch_memory():
                return await self._memory.fetch_user_memory(
                    user_id=request.profile_id,
                    topic=request.custom_learn_prompt or request.custom_talk_prompt or "lesson",
                    subject="english",
                )

            user_profile, memory = await asyncio.gather(fetch_profile(), fetch_memory())

            is_mock_data = user_profile.is_degraded
            if user_profile.child:
                child_age = user_profile.child.age
                child_name = user_profile.child.child_name
            else:
                child_age = None
                child_name = None
            language = user_profile.language_preference or "vi"

            yield _sse({
                "phase": "profile",
                "message": f"Profile fetched: {child_name or 'N/A'}",
                "child_name": child_name,
                "child_age": child_age,
                "language": language,
                "image_url": pipeline_image,
            }, "pipeline")

            artifact_lessons = []
            total_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            for idx, lesson_item in enumerate(request.lessons, start=1):
                yield _sse({
                    "phase": "lesson_started",
                    "index": idx,
                    "total": total_lessons,
                    "title": lesson_item.title,
                    "message": f"Creating artifact {idx}/{total_lessons}: {lesson_item.title}",
                    "image_url": pipeline_image,
                }, "artifact")

                if lesson_item.agent_mode == "learn_agent":
                    artifact_data, lesson_usage = await self._generator.generate_artifact_for_lesson(
                        lesson_title=lesson_item.title,
                        lesson_content=lesson_item.content,
                        lesson_option=lesson_item.option,
                        agent_mode=lesson_item.agent_mode,
                        language=language,
                        child_name=child_name,
                        child_age=child_age,
                        custom_prompt=request.custom_learn_prompt,
                        template_id=lesson_item.template_id,
                    )
                else:
                    artifact_data, lesson_usage = await self._generator.generate_talk_agent_system_task(
                        lesson_content=lesson_item.content,
                        template_id=lesson_item.template_id,
                        language=language,
                        child_name=child_name,
                        child_age=child_age,
                        memory_facts=memory.facts if memory.facts else None,
                        custom_prompt=request.custom_talk_prompt,
                    )

                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    total_token_usage[key] += lesson_usage.get(key, 0)

                lesson_id = f"lesson_{idx:03d}"
                bot_id = AgentBotId[lesson_item.agent_mode.upper()]
                template_id = lesson_item.template_id

                artifact_lesson = {
                    "lesson_id": lesson_id,
                    "bot_id": bot_id,
                    "title": lesson_item.title,
                    "summary": artifact_data.get("summary", ""),
                    "detail_tasks_lesson": artifact_data.get("detail_tasks_lesson", ""),
                    "lesson_json": {
                        "lesson_title": lesson_item.title,
                        "agent_mode": lesson_item.agent_mode,
                        "template_id": template_id,
                        "runtime_bot_template": "PTL_LEARN_GENERIC_BOT_ID" if lesson_item.agent_mode == "learn_agent" else "PTL_TALK_GENERIC_BOT_ID",
                        "system_task_description": artifact_data.get("system_task_description", ""),
                        "card_specs": _enrich_card_specs(artifact_data.get("card_specs", []), lesson_item.exercise_subtypes) if lesson_item.agent_mode == "learn_agent" else artifact_data.get("card_specs", []),
                        "audio_specs": artifact_data.get("audio_specs", []),
                        "checkpoint_specs": _enrich_checkpoint_specs(artifact_data.get("checkpoint_specs", []), lesson_item.exercise_subtypes) if lesson_item.agent_mode == "learn_agent" else artifact_data.get("checkpoint_specs", []),
                    },
                }
                artifact_lessons.append(artifact_lesson)

                yield _sse({
                    "phase": "lesson_complete",
                    "index": idx,
                    "total": total_lessons,
                    "title": lesson_item.title,
                    "lesson": artifact_lesson,
                    "image_url": pipeline_image,
                }, "artifact")

            elapsed_ms = int((time.monotonic() - start) * 1000)

            data = {
                "rejected": False,
                "reason_code": None,
                "reason": "",
                "content": "Created lessons successfully",
                "lessons": artifact_lessons,
                "metadata": {
                    "request_id": request_id,
                    "profile_id": request.profile_id,
                    "child_name": child_name,
                    "child_age": child_age,
                    "language": language,
                    "model_used": "gpt-4o + gpt-4.1",
                    "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                    "processing_time_ms": elapsed_ms,
                    "content_safety_passed": True,
                    "pipeline_version": "2.0.0",
                    "is_mock_data": is_mock_data,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expert_discussion_log": "",
                    "vision_extracted_text": None,
                    "token_usage": total_token_usage,
                    "cost_usd": estimate_cost(
                        model="gpt-4.1",
                        prompt_tokens=total_token_usage["prompt_tokens"],
                        completion_tokens=total_token_usage["completion_tokens"],
                    ),
                },
            }

            yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            logger.warning("pipeline.artifact.cancelled", log_type="job", feature="ARTIFACT", request_id=request_id)
            raise
        except Exception as exc:
            logger.exception("pipeline.artifact.error", log_type="job", feature="ARTIFACT", request_id=request_id, error=str(exc))
            yield _sse({"phase": "error", "message": str(exc)}, "error")
            yield "data: [DONE]\n\n"

    async def generate_regenerate(self, request: RegenerateLessonRequest, *, request_id: str) -> dict:
        start = time.monotonic()

        parent_config = request.optional_parent_config
        subject = parent_config.subject if parent_config else "english"
        purpose = parent_config.purpose if parent_config else "review"
        custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)

        async def fetch_profile():
            return await self._profile.fetch_profile(
                profile_id=request.profile_id,
                token=request.profile_api_token if hasattr(request, 'profile_api_token') else None,
            )

        async def fetch_memory():
            return await self._memory.fetch_user_memory(
                user_id=request.profile_id,
                topic=custom_prompt or subject,
                subject=subject,
            )

        async def extract_vision():
            if request.image_urls:
                return await self._extraction.extract_from_images(
                    image_urls=request.image_urls,
                    subject_hint=subject,
                )
            return ""

        user_profile, memory, image_description = await asyncio.gather(
            fetch_profile(),
            fetch_memory(),
            extract_vision(),
        )

        is_mock_data = user_profile.is_degraded
        if user_profile.child:
            child_age = user_profile.child.age or (parent_config.child_age if parent_config else None)
            child_name = user_profile.child.child_name or (parent_config.child_name if parent_config else None)
        else:
            child_age = parent_config.child_age if parent_config else None
            child_name = parent_config.child_name if parent_config else None

        language = (parent_config.language if parent_config and parent_config.language else None) or user_profile.language_preference or "vi"

        extracted_content = {
            "raw_text": image_description or custom_prompt or "",
            "topic_detected": custom_prompt or subject,
            "subject_detected": subject,
        }

        expert_log, lesson_plan = await self._generator.generate_regenerate_lesson(
            original_lesson=request.lessons_needing_regeneration,
            extracted_content=extracted_content,
            subject=subject,
            purpose=purpose,
            language=language,
            memory_facts=memory.facts if memory.facts else None,
            parent_notes=custom_prompt,
            child_age=child_age,
            child_name=child_name,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None

        lesson_plan = _transform_lesson_plan(
            lesson_plan,
            child_name=child_name,
            child_age=child_age,
            language=language,
            favorite_movie=None,
            dynamic_memory=dynamic_memory,
        )

        lesson_plan = _enforce_single_lesson(lesson_plan)

        data = {
            **lesson_plan,
            "metadata": {
                "request_id": request_id,
                "profile_id": request.profile_id,
                "parent_lesson_id": request.lessons_needing_regeneration.lesson_id,
                "child_name": child_name,
                "child_age": child_age,
                "language": language,
                "model_used": "gpt-4o + gpt-4.1",
                "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                "processing_time_ms": elapsed_ms,
                "content_safety_passed": True,
                "pipeline_version": "2.0.0-regen",
                "is_mock_data": is_mock_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expert_discussion_log": expert_log,
                "vision_extracted_text": image_description,
            },
        }

        return {"request_id": request_id, "status": "success", "data": data}

    # ------------------------------------------------------------------
    # SSE Streaming mode
    # ------------------------------------------------------------------
    async def stream_generate(
        self,
        request: GenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        """Stream lesson generation with SSE."""
        start = time.monotonic()

        try:
            parent_config = request.optional_parent_config
            subject = parent_config.subject if parent_config else "english"
            purpose = parent_config.purpose if parent_config else "review"
            custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)

            pipeline_image = getattr(self._settings, "pipeline_robot_image_url", "") if self._settings else ""
            yield _sse({"phase": "started", "request_id": request_id, "message": "Pika is analyzing request...", "image_url": pipeline_image}, "pipeline")

            async def fetch_profile():
                return await self._profile.fetch_profile(
                    profile_id=request.profile_id,
                    token=request.profile_api_token if hasattr(request, 'profile_api_token') else None,
                )

            async def fetch_memory():
                return await self._memory.fetch_user_memory(
                    user_id=request.profile_id,
                    topic=custom_prompt or subject,
                    subject=subject,
                )

            async def extract_vision():
                if request.image_urls:
                    return await self._extraction.extract_from_images(
                        image_urls=request.image_urls,
                        subject_hint=subject,
                    )
                return ""

            user_profile, memory, image_description = await asyncio.gather(
                fetch_profile(),
                fetch_memory(),
                extract_vision(),
            )

            is_mock_data = user_profile.is_degraded

            if user_profile.child:
                child_age = user_profile.child.age or (parent_config.child_age if parent_config else None)
                child_name = user_profile.child.child_name or (parent_config.child_name if parent_config else None)
            else:
                child_age = parent_config.child_age if parent_config else None
                child_name = parent_config.child_name if parent_config else None

            language = (parent_config.language if parent_config and parent_config.language else None) or user_profile.language_preference or "vi"

            yield _sse({
                "phase": "profile",
                "message": f"Profile fetched: {user_profile.child.child_name if user_profile.child else 'N/A'}",
                "child_name": user_profile.child.child_name if user_profile.child else None,
                "child_age": user_profile.child.age if user_profile.child else None,
                "language": user_profile.language_preference,
                "image_url": pipeline_image,
            }, "pipeline")

            yield _sse({
                "phase": "memory",
                "message": "Pika is searching for relevant information...",
                "facts": [f.text for f in memory.facts],
                "image_url": pipeline_image,
            }, "pipeline")

            if request.image_urls:
                preview = image_description[:100] + "..." if len(image_description) > 100 else image_description
                yield _sse({
                    "phase": "vision_complete",
                    "message": f"Extracted: {preview}",
                    "vision_extracted_text": image_description,
                    "image_url": pipeline_image,
                }, "pipeline")

            extracted_content = {
                "raw_text": image_description,
                "topic_detected": custom_prompt or subject,
                "subject_detected": subject,
            }

            expert_log_parts: list[str] = []
            lesson_plan: dict = {}
            final_editor_content = ""

            async for event_type, content in self._generator.stream_generate_lesson(
                extracted_content=extracted_content,
                subject=subject,
                purpose=purpose,
                language=language,
                memory_facts=memory.facts if memory.facts else None,
                parent_notes=custom_prompt,
                child_age=child_age,
                child_name=child_name,
            ):
                if event_type.startswith("thinking:"):
                    expert_key = event_type.split(":", 1)[1]
                    expert_log_parts.append(content)

                    if expert_key == "final_editor":
                        final_editor_content += content + "\n"
                    else:
                        avatar_attr = self.EXPERT_AVATARS.get(expert_key, "")
                        avatar_url = getattr(self._settings, avatar_attr, "") if self._settings else ""
                        yield _sse({
                            "phase": "thinking",
                            "expert": expert_key,
                            "expert_name": expert_key.replace("_", " ").title(),
                            "content": content,
                            "avatar_url": avatar_url,
                        }, "thinking")

                elif event_type == "json_complete":
                    lesson_plan = json.loads(content)

                    clean_content = final_editor_content
                    json_start = clean_content.find("{")
                    if json_start != -1:
                        clean_content = clean_content[:json_start]
                    clean_content = clean_content.strip()

                    avatar_attr = self.EXPERT_AVATARS.get("final_editor", "")
                    avatar_url = getattr(self._settings, avatar_attr, "") if self._settings else ""
                    yield _sse({
                        "phase": "thinking",
                        "expert": "final_editor",
                        "expert_name": "Final Editor",
                        "content": lesson_plan.get("content", ""),
                        "metadata": lesson_plan,
                        "avatar_url": avatar_url,
                    }, "thinking")

                elif event_type == "error":
                    yield _sse({"phase": "error", "message": content}, "error")
                    yield "data: [DONE]\n\n"
                    return

            elapsed_ms = int((time.monotonic() - start) * 1000)
            dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None

            lesson_plan = _transform_lesson_plan(
                lesson_plan,
                child_name=child_name,
                child_age=child_age,
                language=language,
                favorite_movie=None,
                dynamic_memory=dynamic_memory,
            )

            data = {
                **lesson_plan,
                "metadata": {
                    "request_id": request_id,
                    "profile_id": request.profile_id,
                    "child_name": child_name,
                    "child_age": child_age,
                    "language": language,
                    "model_used": "gpt-4o + gpt-4.1",
                    "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                    "processing_time_ms": elapsed_ms,
                    "content_safety_passed": True,
                    "pipeline_version": "2.0.0",
                    "is_mock_data": is_mock_data,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expert_discussion_log": "\n".join(expert_log_parts),
                    "vision_extracted_text": image_description,
                },
            }

            yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            logger.warning("pipeline.cancelled", log_type="job", feature="LESSON", request_id=request_id)
            raise
        except Exception as exc:
            logger.exception("pipeline.error", log_type="job", feature="LESSON", request_id=request_id, error=str(exc))
            yield _sse({"phase": "error", "message": str(exc)}, "error")
            yield "data: [DONE]\n\n"

    async def stream_regenerate(
        self,
        request: RegenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        """Stream lesson regeneration with SSE."""
        start = time.monotonic()

        try:
            parent_config = request.optional_parent_config
            subject = parent_config.subject if parent_config else "english"
            purpose = parent_config.purpose if parent_config else "review"
            custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)

            pipeline_image = getattr(self._settings, "pipeline_robot_image_url", "") if self._settings else ""
            yield _sse({"phase": "started", "request_id": request_id, "message": "Pika is regenerating lesson...", "image_url": pipeline_image}, "pipeline")

            async def fetch_profile():
                return await self._profile.fetch_profile(
                    profile_id=request.profile_id,
                    token=request.profile_api_token if hasattr(request, 'profile_api_token') else None,
                )

            async def fetch_memory():
                return await self._memory.fetch_user_memory(
                    user_id=request.profile_id,
                    topic=custom_prompt or subject,
                    subject=subject,
                )

            async def extract_vision():
                if request.image_urls:
                    return await self._extraction.extract_from_images(
                        image_urls=request.image_urls,
                        subject_hint=subject,
                    )
                return ""

            user_profile, memory, image_description = await asyncio.gather(
                fetch_profile(),
                fetch_memory(),
                extract_vision(),
            )

            is_mock_data = user_profile.is_degraded
            if user_profile.child:
                child_age = user_profile.child.age or (parent_config.child_age if parent_config else None)
                child_name = user_profile.child.child_name or (parent_config.child_name if parent_config else None)
            else:
                child_age = parent_config.child_age if parent_config else None
                child_name = parent_config.child_name if parent_config else None

            language = (parent_config.language if parent_config and parent_config.language else None) or user_profile.language_preference or "vi"

            yield _sse({
                "phase": "profile",
                "message": f"Profile fetched: {user_profile.child.child_name if user_profile.child else 'N/A'}",
                "child_name": user_profile.child.child_name if user_profile.child else None,
                "child_age": user_profile.child.age if user_profile.child else None,
                "language": user_profile.language_preference,
                "image_url": pipeline_image,
            }, "pipeline")

            yield _sse({
                "phase": "memory",
                "message": "Pika is searching for relevant information...",
                "facts": [f.text for f in memory.facts],
                "image_url": pipeline_image,
            }, "pipeline")

            extracted_content = {
                "raw_text": image_description,
                "topic_detected": custom_prompt or subject,
                "subject_detected": subject,
            }

            expert_log_parts: list[str] = []
            lesson_plan: dict = {}

            async for event_type, content in self._generator.stream_regenerate_lesson(
                original_lesson=request.lessons_needing_regeneration,
                extracted_content=extracted_content,
                subject=subject,
                purpose=purpose,
                language=language,
                memory_facts=memory.facts if memory.facts else None,
                parent_notes=custom_prompt,
                child_age=child_age,
                child_name=child_name,
            ):
                if event_type.startswith("thinking:"):
                    expert_key = event_type.split(":", 1)[1]
                    expert_log_parts.append(content)
                    avatar_attr = self.EXPERT_AVATARS.get(expert_key, "")
                    avatar_url = getattr(self._settings, avatar_attr, "") if self._settings else ""
                    yield _sse({
                        "phase": "thinking",
                        "expert": expert_key,
                        "expert_name": expert_key.replace("_", " ").title(),
                        "content": content,
                        "avatar_url": avatar_url,
                    }, "thinking")

                elif event_type == "json_complete":
                    lesson_plan = json.loads(content)

                elif event_type == "error":
                    yield _sse({"phase": "error", "message": content}, "error")
                    yield "data: [DONE]\n\n"
                    return

            elapsed_ms = int((time.monotonic() - start) * 1000)
            dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None

            lesson_plan = _transform_lesson_plan(
                lesson_plan,
                child_name=child_name,
                child_age=child_age,
                language=language,
                favorite_movie=None,
                dynamic_memory=dynamic_memory,
            )

            lesson_plan = _enforce_single_lesson(lesson_plan)

            data = {
                **lesson_plan,
                "metadata": {
                    "request_id": request_id,
                    "profile_id": request.profile_id,
                    "parent_lesson_id": request.lessons_needing_regeneration.lesson_id,
                    "child_name": child_name,
                    "child_age": child_age,
                    "language": language,
                    "model_used": "gpt-4o + gpt-4.1",
                    "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                    "processing_time_ms": elapsed_ms,
                    "content_safety_passed": True,
                    "pipeline_version": "2.0.0-regen",
                    "is_mock_data": is_mock_data,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expert_discussion_log": "\n".join(expert_log_parts),
                    "vision_extracted_text": image_description,
                },
            }

            yield _sse({"phase": "complete", "data": data, "image_url": pipeline_image}, "pipeline")
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("pipeline.regenerate.error", request_id=request_id, error=str(exc))
            yield _sse({"phase": "error", "message": str(exc)}, "error")
            yield "data: [DONE]\n\n"
