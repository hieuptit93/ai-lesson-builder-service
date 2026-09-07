"""LessonPipeline: orchestrates Memory -> Vision -> Lesson Generation.

Supports both sync (JSON) and SSE streaming modes.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import structlog
from langfuse import observe

from app.api.v3.schemas.artifact_request import GenerateArtifactRequest
from app.api.v1.schemas.lesson_request import GenerateLessonRequest, RegenerateLessonRequest
from app.core.config import Settings
from app.core.enums import AgentBotId
from app.domains.lesson_generator.application.services.generator_service import GeneratorService
from app.domains.lesson_generator.application.services.prompt_builder import (
    USER_PROFILE_PROMPT_TEMPLATE,
    get_langfuse_prompt,
    load_prompt_file,
    _get_language_prompt,
    _get_task_base_on_language_prompt,
)
from app.domains.memory.application.services.memory_service import MemoryService
from app.domains.memory.domain.entities import UserMemory
from app.domains.profile.application.services.profile_service import ProfileService
from app.domains.profile.domain.entities import UserProfile
from app.core.exceptions import AIServiceError, MissingImageUrlsError, UnsafeContentError, VisionExtractionError
from app.domains.vision_extract.application.services.extraction_service import ExtractionService
from app.utils.cost import cache_savings, estimate_cost
from app.utils.lesson_validation import summarize_reports, validate_lessons

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
    "ptl_learn_vocab_flashcard_v1": "Từ vựng",
    "ptl_learn_phonics_pronunciation_v1": "Phát âm",
    "ptl_learn_exercise_solver_v1": "Giải bài tập",
    "ptl_learn_sentence_pattern_practice_v1": "Luyện mẫu câu",
    "ptl_learn_reading_comprehension_v1": "Đọc hiểu",
    "ptl_talk_roleplay_v1": "Nhập vai hội thoại",
    "ptl_talk_speaking_presentation_v1": "Thuyết trình",
    "ptl_talk_storytelling_v1": "Kể chuyện sáng tạo",
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

    # Try to fetch prompts from Langfuse, fallback to the bundled prompts/ copy
    context_style = get_langfuse_prompt("context_style_guideline_prompt") or load_prompt_file(
        "context_style_guideline_prompt"
    )

    # user_profile_prompt has no bundled copy on purpose: it is a tiny format
    # string owned by the code, not prose worth managing in Langfuse.
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


# Profile/memory sit on the critical path before the vision call (their data
# personalizes generation). Bound them so a slow upstream can't stall lessons -
# both services already fail-open internally, this only guards hangs/timeouts.
_UPSTREAM_SOFT_TIMEOUT_S = 3.0


async def _bounded_profile(coro, profile_id: str) -> "UserProfile":
    try:
        return await asyncio.wait_for(coro, timeout=_UPSTREAM_SOFT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("profile_soft_timeout", profile_id=profile_id, timeout_s=_UPSTREAM_SOFT_TIMEOUT_S)
        return UserProfile(
            user_id=profile_id,
            profile_id=profile_id,
            child=None,
            language_preference="vi",
            raw_data=None,
            is_degraded=True,
            degraded_reason=f"Profile fetch exceeded {_UPSTREAM_SOFT_TIMEOUT_S}s soft timeout",
        )


async def _bounded_memory(coro, profile_id: str) -> "UserMemory":
    try:
        return await asyncio.wait_for(coro, timeout=_UPSTREAM_SOFT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("memory_soft_timeout", profile_id=profile_id, timeout_s=_UPSTREAM_SOFT_TIMEOUT_S)
        return UserMemory(user_id=profile_id, facts=[], query_used="", total_found=0)


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


def _cost_from_usage(model: str, usage: dict | None) -> float:
    """Cost for one call, pricing cached hits and cache writes separately."""
    if not usage:
        return 0.0
    return estimate_cost(
        model=model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cached_tokens=usage.get("cached_tokens", 0),
        cache_write_tokens=usage.get("cache_write_tokens", 0),
    )


def _cache_savings_from_usage(model: str, usage: dict | None) -> float:
    """Net USD the prompt cache saved on one call (negative on a write-only call)."""
    if not usage:
        return 0.0
    return cache_savings(
        model,
        usage.get("cached_tokens", 0),
        usage.get("cache_write_tokens", 0),
    )


def _validate_and_log(lessons: list, *, request_id: str, flow: str) -> dict:
    """Run cross-field validation, log findings, emit a metric. Never raises.

    Pure-Python check (~1ms/lesson), so it runs inline before the response is
    returned. Errors are structural; warnings are soft signals kept for prompt
    tuning. Neither blocks the response - the summary rides along in metadata
    so callers can decide (and so a future response cache can gate on it).
    """
    try:
        reports = validate_lessons(lessons)
        summary = summarize_reports(reports)

        for report in reports:
            if report.errors:
                logger.warning(
                    "lesson_validation.errors",
                    request_id=request_id,
                    flow=flow,
                    **report.as_log_dict(),
                    messages=[i.message for i in report.errors],
                )
            elif report.warnings:
                logger.info(
                    "lesson_validation.warnings",
                    request_id=request_id,
                    flow=flow,
                    **report.as_log_dict(),
                    messages=[i.message for i in report.warnings],
                )

        try:
            from app.core.metric_events import emit as _metric_emit

            _metric_emit(
                "lesson_validation",
                flow=flow,
                outcome="ok" if summary["all_valid"] else "invalid",
                lessons_checked=summary["lessons_checked"],
                lessons_with_errors=summary["lessons_with_errors"],
                error_count=summary["error_count"],
                warning_count=summary["warning_count"],
            )
        except Exception:  # noqa: BLE001
            pass

        return summary
    except Exception as exc:  # noqa: BLE001
        # Validation must never break generation.
        logger.warning("lesson_validation.failed", request_id=request_id, error=str(exc))
        return {}


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
    @observe(name="lesson_pipeline.generate", capture_input=False, capture_output=True)
    async def generate(self, request: GenerateLessonRequest, *, request_id: str) -> dict:
        """V1 lesson generation - single-call with 5-expert deliberation.

        Optimized flow (Sep 2026):
        1. Use parent_config for child info if available (skip profile wait)
        2. Profile + Memory fetched with bounded timeout (3s) - fallback to defaults
        3. Vision call uses optimized prompt with reasoning_effort=minimal

        Latency breakdown:
        - Profile/Memory: ~1-2s (parallel, bounded)
        - Vision 5-expert: ~10-15s (optimized prompt, minimal reasoning)
        - Total: ~12-17s (vs ~20s before)
        """
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

        # Extract child info from parent_config first (skip profile wait if available)
        config_child_age = parent_config.child_age if parent_config else None
        config_child_name = parent_config.child_name if parent_config else None
        config_language = parent_config.language if parent_config and parent_config.language else None

        # 2-call flow (matches production quality):
        #   Step 1: Profile + Memory + Vision extraction ALL in parallel
        #   Step 2: Generation with full Langfuse prompt (64KB pedagogical rules)
        # Quality comes from the rich generation prompt; speed from parallelism
        # and prompt caching of the static rules (see build_lesson_prompt_split).
        PERSONALIZATION_TIMEOUT = 3.0  # seconds - fallback to defaults if slow

        async def fetch_profile_bounded():
            try:
                return await asyncio.wait_for(
                    self._profile.fetch_profile(
                        profile_id=request.profile_id,
                        token=request.profile_api_token if hasattr(request, 'profile_api_token') else None,
                    ),
                    timeout=PERSONALIZATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("pipeline.profile_timeout", profile_id=request.profile_id, timeout=PERSONALIZATION_TIMEOUT)
                return None

        async def fetch_memory_bounded():
            try:
                return await asyncio.wait_for(
                    self._memory.fetch_user_memory(
                        user_id=request.profile_id,
                        topic=custom_prompt or subject,
                        subject=subject,
                    ),
                    timeout=PERSONALIZATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("pipeline.memory_timeout", profile_id=request.profile_id, timeout=PERSONALIZATION_TIMEOUT)
                return None

        async def extract_vision():
            if request.image_urls:
                raw_text = await self._extraction.extract_from_images(
                    image_urls=request.image_urls,
                    subject_hint=subject,
                )
                return raw_text
            return None

        # Run ALL THREE in parallel - Vision dominates (~8-13s), Profile/Memory (~0.3s)
        user_profile, memory, vision_raw_text = await asyncio.gather(
            fetch_profile_bounded(),
            fetch_memory_bounded(),
            extract_vision(),
        )

        # Use config values first, fall back to profile if available
        is_mock_data = user_profile.is_degraded if user_profile else True

        if user_profile and user_profile.child:
            child_age = user_profile.child.age or config_child_age
            child_name = user_profile.child.child_name or config_child_name
        else:
            child_age = config_child_age
            child_name = config_child_name

        language = config_language or (user_profile.language_preference if user_profile else None) or "vi"

        # Step 2: Generation with full production prompt (5-expert deliberation)
        if request.image_urls:
            # topic_detected stays as subject - the model infers topic from
            # RAW_TEXT. custom_prompt goes to PARENT_SECTION via parent_notes;
            # passing it as TOPIC creates a false topic-vs-image conflict that
            # trips the prompt's not_found gate.
            extracted_content = {
                "raw_text": vision_raw_text or "",
                "topic_detected": subject,
                "subject_detected": subject,
            }
            expert_log, lesson_plan = await self._generator.generate_lesson(
                extracted_content=extracted_content,
                subject=subject,
                purpose=purpose,
                language=language,
                memory_facts=memory.facts if memory and memory.facts else None,
                parent_notes=custom_prompt,
                child_age=child_age,
                child_name=child_name,
            )
            token_usage = {}
        else:
            # No images: fall back to text-only generation
            extracted_content = {
                "raw_text": custom_prompt or "",
                "topic_detected": custom_prompt or subject,
                "subject_detected": subject,
            }
            expert_log, lesson_plan = await self._generator.generate_lesson(
                extracted_content=extracted_content,
                subject=subject,
                purpose=purpose,
                language=language,
                memory_facts=memory.facts if memory and memory.facts else None,
                parent_notes=custom_prompt,
                child_age=child_age,
                child_name=child_name,
            )
            token_usage = {}

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Check for rejected content
        if lesson_plan.get("rejected"):
            logger.warning(
                "pipeline.rejected",
                log_type="job",
                feature="LESSON",
                request_id=request_id,
                reason_code=lesson_plan.get("reason_code"),
                reason=lesson_plan.get("reason"),
            )
            return {
                "request_id": request_id,
                "status": "rejected",
                "data": {
                    "rejected": True,
                    "reason_code": lesson_plan.get("reason_code", "content_rejected"),
                    "reason": lesson_plan.get("reason", "Content was rejected"),
                    "lessons": [],
                    "metadata": {
                        "request_id": request_id,
                        "profile_id": request.profile_id,
                        "processing_time_ms": elapsed_ms,
                        "content_safety_passed": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                },
            }

        dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None

        lesson_plan = _transform_lesson_plan(
            lesson_plan,
            child_name=child_name,
            child_age=child_age,
            language=language,
            favorite_movie=None,
            dynamic_memory=dynamic_memory,
        )

        # Calculate cost for single-call model
        vision_model = self._settings.openai_vision_model if self._settings else "gpt-5.6-terra"
        cost_usd = _cost_from_usage(vision_model, token_usage)

        data = {
            **lesson_plan,
            "metadata": {
                "request_id": request_id,
                "profile_id": request.profile_id,
                "child_name": child_name,
                "child_age": child_age,
                "language": language,
                "model_used": vision_model,  # Single model now
                "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                "processing_time_ms": elapsed_ms,
                "content_safety_passed": True,
                "pipeline_version": "2.1.0",  # Bumped for single-call optimization
                "is_mock_data": is_mock_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expert_discussion_log": expert_log,
                "vision_extracted_text": lesson_plan.get("content", ""),  # From 5-expert extraction
                "usage": token_usage,
                "cost_usd": cost_usd,
                "cached_tokens": token_usage.get("cached_tokens", 0),
                "cache_write_tokens": token_usage.get("cache_write_tokens", 0),
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
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            cost_usd=cost_usd,
        )

        return {"request_id": request_id, "status": "success", "data": data}

    @observe(name="lesson_pipeline.generate_v3", capture_input=False, capture_output=False)
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

            # v3/lessons/generate returns suggested_lessons ONLY - no personalization needed.
            # Run guardrail + vision + profile ALL IN PARALLEL for maximum speed.
            # Profile is fetched for metadata/logging but NOT used for generation.

            async def run_vision():
                return await self._extraction.extract_suggestions_v3(
                    image_urls=request.image_urls,
                    subject_hint=subject,
                    custom_prompt=custom_prompt,
                )

            # All three run concurrently - vision doesn't wait for anything
            guardrail_result, vision_result, user_profile = await asyncio.gather(
                run_guardrail(),
                run_vision(),
                _bounded_profile(fetch_profile(), request.profile_id),
                return_exceptions=True,
            )

            # Handle guardrail result
            guardrail_detail = ""
            if isinstance(guardrail_result, Exception):
                logger.warning("guardrail_exception", error=str(guardrail_result))
            else:
                guardrail_detail = guardrail_result or ""

            # Handle vision result
            if isinstance(vision_result, Exception):
                raise vision_result
            extracted_result, token_usage = vision_result

            # Handle profile result (for logging only)
            if isinstance(user_profile, Exception):
                logger.warning("profile_exception", error=str(user_profile))
                user_profile = UserProfile(
                    user_id=request.profile_id,
                    profile_id=request.profile_id,
                    child=None,
                    language_preference="vi",
                    raw_data=None,
                    is_degraded=True,
                    degraded_reason=str(user_profile),
                )
            # Check guardrail after gather - vision already ran (optimistic execution)
            if guardrail_detail:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.warning(
                    "guardrail_rejected",
                    request_id=request_id,
                    reason=guardrail_detail,
                )
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "data": {
                        "detail": {"code": "unsafe_content", "message": guardrail_detail},
                        "metadata": {
                            "language": language,
                            "processing_time_ms": elapsed_ms,
                            "content_safety_passed": False,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "token_usage": None,
                            "cost_usd": 0.0,
                        },
                        "suggested_lessons": [],
                    },
                }

            # extracted_result and token_usage already set from gather

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
                        "detail": {},
                        "metadata": {
                            "language": language,
                            "processing_time_ms": elapsed_ms,
                            "content_safety_passed": False,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "token_usage": token_usage,
                            "cost_usd": 0.0,
                        },
                        "suggested_lessons": [],
                    },
                }

            # v3/lessons/generate returns suggested_lessons (not full lessons)
            # extracted_result is a dict with "suggested_lessons" array
            suggested_lessons_list = extracted_result.get("suggested_lessons", [])

            if not suggested_lessons_list:
                # Model found no teachable content
                logger.warning("vision_v3_empty_suggestions", request_id=request_id)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "data": {
                        "detail": {},
                        "metadata": {
                            "language": language,
                            "processing_time_ms": elapsed_ms,
                            "content_safety_passed": True,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "token_usage": token_usage,
                            "cost_usd": 0.0,
                        },
                        "suggested_lessons": [],
                    },
                }

            # Enrich options for each suggestion
            for suggestion in suggested_lessons_list:
                suggestion["options"] = _enrich_options(suggestion.get("options", []))

            is_mock_data = user_profile.is_degraded
            if user_profile.child:
                child_age = user_profile.child.age
                child_name = user_profile.child.child_name
            else:
                child_age = parent_config.child_age if parent_config else None
                child_name = parent_config.child_name if parent_config else None

            language = (parent_config.language if parent_config and parent_config.language else None) or user_profile.language_preference or "vi"

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # v3 runs on the suggestions model, so cost must be priced with it.
            vision_model = self._settings.openai_suggestions_model if self._settings else "gpt-5.6-terra"

            # Return suggested_lessons format matching source system
            # NOTE: No finally_prompt_agent here - that's created by generate_artifact
            data = {
                "detail": {},
                "metadata": {
                    "language": language,
                    "child_name": child_name,
                    "child_age": child_age,
                    "is_mock_data": is_mock_data,
                    "processing_time_ms": elapsed_ms,
                    "content_safety_passed": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "token_usage": token_usage,
                    "cost_usd": _cost_from_usage(vision_model, token_usage),
                    "cached_tokens": (token_usage or {}).get("cached_tokens", 0),
                    "cache_write_tokens": (token_usage or {}).get("cache_write_tokens", 0),
                    "cache_savings_usd": _cache_savings_from_usage(vision_model, token_usage),
                },
                "suggested_lessons": suggested_lessons_list,
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
                    "detail": {"code": exc.error_code, "message": exc.message},
                    "suggested_lessons": [],
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
                    "detail": {"code": "internal_error", "message": str(exc)},
                    "suggested_lessons": [],
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

    @observe(name="lesson_pipeline.stream_generate_v3", capture_input=False, capture_output=False)
    async def stream_generate_v3(
        self,
        request: GenerateLessonRequest,
        *,
        request_id: str,
        delay: float = 0.0,
        use_full_prompt: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream v3 lesson generation (vision -> suggested_lessons) with SSE.

        Args:
            use_full_prompt: If True, use full prompt with D-steps (for v1 optimization ~10s).
                            If False, use Langfuse lightweight prompt (for v3).
        """
        start = time.monotonic()

        try:
            parent_config = request.optional_parent_config
            subject = parent_config.subject if parent_config else "english"
            custom_prompt = request.custom_prompt if request.custom_prompt is not None else (parent_config.custom_prompt if parent_config else None)

            pipeline_image = getattr(self._settings, "pipeline_robot_image_url", "") if self._settings else ""
            yield _sse({"phase": "started", "request_id": request_id, "message": "Pika đang phân tích yêu cầu...", "image_url": pipeline_image}, "pipeline")

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

            # OPTIMISTIC EXECUTION: guardrail starts immediately and overlaps
            # the (bounded) profile/memory fetch AND the vision stream.
            guardrail_task = asyncio.create_task(run_guardrail())

            user_profile, memory = await asyncio.gather(
                _bounded_profile(fetch_profile(), request.profile_id),
                _bounded_memory(fetch_memory(), request.profile_id),
            )

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
                "message": f"Đã lấy thông tin bé: {child_name or 'N/A'}",
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

            # REAL STREAMING: lessons are pushed to the client the moment each
            # one's JSON completes, instead of waiting for the full response.
            #
            # v1 (use_full_prompt=True): 2-call flow matching the sync path -
            #   vision extraction (fast model) overlaps the guardrail, then the
            #   generation call streams lessons using the full cached
            #   production prompt (same quality as sync generate()).
            # v3 (use_full_prompt=False): single vision stream, unchanged.
            purpose = parent_config.purpose if parent_config else "review"
            language_resolved = (
                (parent_config.language if parent_config and parent_config.language else None)
                or user_profile.language_preference
                or "vi"
            )

            vision_task = None
            if use_full_prompt:
                vision_task = asyncio.create_task(
                    self._extraction.extract_from_images(
                        image_urls=request.image_urls,
                        subject_hint=subject,
                    )
                )
                stream_gen = None
                first_event_task = None
            else:
                stream_gen = self._extraction.extract_from_images_v3_stream(
                    image_urls=request.image_urls,
                    subject_hint=subject,
                    custom_prompt=custom_prompt,
                    personalization_context=personalization_context,
                    use_full_prompt=use_full_prompt,
                )
                first_event_task = asyncio.create_task(stream_gen.__anext__())

            guardrail_detail = await guardrail_task
            if guardrail_detail:
                if vision_task:
                    vision_task.cancel()
                if first_event_task:
                    first_event_task.cancel()
                if stream_gen:
                    await stream_gen.aclose()
                elapsed_ms = int((time.monotonic() - start) * 1000)
                language = (parent_config.language if parent_config and parent_config.language else None) or "vi"
                data = {
                    "detail": {"code": "unsafe_content", "message": guardrail_detail},
                    "suggested_lessons": [],
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

            # v1: wait for vision extraction, then start the streaming
            # generation call (full cached production prompt).
            if use_full_prompt:
                vision_raw_text = await vision_task
                yield _sse({
                    "phase": "vision_complete",
                    "message": "Đã phân tích hình ảnh xong",
                    "image_url": pipeline_image,
                }, "pipeline")
                stream_gen = self._generator.stream_generate_lesson_v2(
                    extracted_content={
                        "raw_text": vision_raw_text or "",
                        "topic_detected": subject,
                        "subject_detected": subject,
                    },
                    subject=subject,
                    purpose=purpose,
                    language=language_resolved,
                    memory_facts=memory.facts if memory.facts else None,
                    parent_notes=custom_prompt,
                    child_age=child_age,
                    child_name=child_name,
                )
                first_event_task = asyncio.create_task(stream_gen.__anext__())

            # Consume the stream. "lesson_ready" events are previews
            # (no finally_prompt_agent yet); the final "complete" payload
            # carries the authoritative fully-transformed lessons.
            extracted_result: dict = {}
            token_usage: dict = {}
            expert_log_parts: list[str] = []
            lesson_index = 0
            event = await first_event_task
            while True:
                event_type, payload = event
                if event_type == "lesson":
                    lesson_index += 1
                    preview = dict(payload)
                    preview["lesson_id"] = f"lesson_{lesson_index:03d}"
                    preview["options"] = _enrich_options(preview.get("options", []))
                    yield _sse({
                        "phase": "lesson_ready",
                        "index": lesson_index,
                        "lesson": preview,
                        "image_url": pipeline_image,
                    }, "lesson")
                elif event_type.startswith("thinking:"):
                    expert_key = event_type.split(":", 1)[1]
                    expert_log_parts.append(payload)
                    # final_editor only emits JSON, so its event is synthesized
                    # after the lessons are known (see below).
                    if expert_key != "final_editor":
                        avatar_attr = self.EXPERT_AVATARS.get(expert_key, "")
                        avatar_url = getattr(self._settings, avatar_attr, "") if self._settings else ""
                        yield _sse({
                            "phase": "thinking",
                            "expert": expert_key,
                            "expert_name": expert_key.replace("_", " ").title(),
                            "content": payload,
                            "avatar_url": avatar_url,
                        }, "thinking")
                elif event_type == "complete":
                    extracted_result, token_usage = payload
                    break
                try:
                    event = await stream_gen.__anext__()
                except StopAsyncIteration:
                    break

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
                reason_code = extracted_result.get("reason_code", "content_rejected")
                reason_msg = extracted_result.get("reason", "Content was rejected")
                data = {
                    "detail": {"code": reason_code, "message": reason_msg},
                    "suggested_lessons": [],
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
                # Model found no teachable content but did not reject - treat as empty result
                logger.warning("vision_v3_empty_lessons", request_id=request_id)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                language = (parent_config.language if parent_config and parent_config.language else None) or "vi"
                data = {
                    "detail": {"code": "no_educational_content", "message": "Không tìm thấy nội dung học tập rõ ràng trong ảnh"},
                    "suggested_lessons": [],
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

            if request.image_urls and not use_full_prompt:
                # v1 already emitted vision_complete right after extraction
                yield _sse({
                    "phase": "vision_complete",
                    "message": "Đã phân tích hình ảnh xong",
                    "image_url": pipeline_image,
                }, "pipeline")

            # Cross-field validation runs on the RAW model output, before our
            # own transform touches it - that measures prompt quality, not
            # post-processing. Only the v1 flow has the fields to check
            # (summary / detail_tasks_lesson / prompt_agent); v3 suggestions
            # carry title+content only and are validated after generate_artifact.
            validation_summary: dict = {}
            if use_full_prompt:
                validation_summary = _validate_and_log(
                    extracted_result.get("lessons", []),
                    request_id=request_id,
                    flow="v1_stream",
                )

            # v1 needs full transform (lesson_id, finally_prompt_agent, etc)
            # v3 needs raw lessons from vision (agent_mode, content, options, title only)
            if use_full_prompt:
                # v1: Transform lessons with personalization
                dynamic_memory = "\n".join(f"- {f.text}" for f in memory.facts) if memory.facts else None
                transformed_result = _transform_lesson_plan(
                    {"lessons": extracted_result.get("lessons", [])},
                    child_name=child_name,
                    child_age=child_age,
                    language=language,
                    favorite_movie=None,
                    dynamic_memory=dynamic_memory,
                )
                # Strip fields not needed for v1
                FIELDS_TO_REMOVE = {"agent_mode", "options", "content"}
                for lesson in transformed_result.get("lessons", []):
                    for field in FIELDS_TO_REMOVE:
                        lesson.pop(field, None)
            else:
                # v3: Use raw lessons with only required fields (agent_mode, content, options, title)
                raw_lessons = extracted_result.get("suggested_lessons", []) or extracted_result.get("lessons", [])
                V3_FIELDS_TO_KEEP = {"agent_mode", "content", "options", "title"}
                filtered_lessons = [
                    {k: v for k, v in lesson.items() if k in V3_FIELDS_TO_KEEP}
                    for lesson in raw_lessons
                ]
                transformed_result = {"lessons": filtered_lessons}

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # v1's token_usage comes from the GENERATION call (2-call flow);
            # v3's comes from the suggestions vision call.
            if self._settings:
                vision_model = (
                    self._settings.openai_lesson_model
                    if use_full_prompt
                    else self._settings.openai_suggestions_model
                )
            else:
                vision_model = "gpt-4.1" if use_full_prompt else "gpt-5.6-terra"

            if use_full_prompt:
                # v1: the real image description from the extraction call
                vision_extracted_text = vision_raw_text or ""
            else:
                # v3: no separate extraction - approximate from lesson summaries
                lessons_list = extracted_result.get("suggested_lessons", []) or extracted_result.get("lessons", [])
                vision_text_parts = [lesson.get("summary", "") for lesson in lessons_list if lesson.get("summary")]
                vision_extracted_text = "\n".join(vision_text_parts) if vision_text_parts else ""

            # Build response data - different format for v1 vs v3
            lessons_data = transformed_result.get("lessons", [])

            metadata = {
                "request_id": request_id,
                "profile_id": request.profile_id,
                "language": language,
                "child_name": child_name,
                "child_age": child_age,
                "memory_facts_used": [f.text for f in memory.facts] if memory.facts else [],
                "processing_time_ms": elapsed_ms,
                "content_safety_passed": True,
                "pipeline_version": "2.0.0",
                "is_mock_data": is_mock_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_used": vision_model,
                "expert_discussion_log": "\n".join(expert_log_parts),
                "vision_extracted_text": vision_extracted_text,
                "usage": token_usage,
                "cost_usd": _cost_from_usage(vision_model, token_usage),
                "cached_tokens": (token_usage or {}).get("cached_tokens", 0),
                "cache_write_tokens": (token_usage or {}).get("cache_write_tokens", 0),
                "cache_savings_usd": _cache_savings_from_usage(vision_model, token_usage),
                "validation": validation_summary,
            }

            if use_full_prompt:
                # v1 format: uses "lessons" key with full metadata
                data = {
                    "rejected": extracted_result.get("rejected", False),
                    "reason_code": extracted_result.get("reason_code"),
                    "reason": extracted_result.get("reason", ""),
                    "content": "Đã tạo bài học thành công",
                    "lessons": lessons_data,
                    "metadata": metadata,
                }
            else:
                # v3 format: uses "suggested_lessons" key, simpler structure
                data = {
                    "rejected": extracted_result.get("rejected", False),
                    "reason_code": extracted_result.get("reason_code"),
                    "reason": extracted_result.get("reason", ""),
                    "detail": "",
                    "suggested_lessons": lessons_data,
                    "metadata": metadata,
                }

            if use_full_prompt:
                avatar_attr = self.EXPERT_AVATARS.get("final_editor", "")
                avatar_url = getattr(self._settings, avatar_attr, "") if self._settings else ""
                yield _sse({
                    "phase": "thinking",
                    "expert": "final_editor",
                    "expert_name": "Final Editor",
                    "content": data["content"],
                    "metadata": {"lesson_count": len(lessons_data)},
                    "avatar_url": avatar_url,
                }, "thinking")

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

    @observe(name="lesson_pipeline.generate_artifact_v3", capture_input=False, capture_output=False)
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

        total_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0, "total_tokens": 0}

        # OPTIMIZATION: Generate all lessons in PARALLEL instead of sequential
        async def generate_single_lesson(idx: int, lesson_item):
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

            return {
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
                "_usage": lesson_usage,
            }

        # Run all lessons in parallel
        results = await asyncio.gather(
            *[generate_single_lesson(idx, lesson) for idx, lesson in enumerate(request.lessons, start=1)],
            return_exceptions=True,
        )

        # Process results
        artifact_lessons = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("lesson_generation_failed", error=str(result))
                continue
            # Extract and accumulate token usage
            lesson_usage = result.pop("_usage", {})
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens", "total_tokens"):
                total_token_usage[key] += lesson_usage.get(key, 0)
            artifact_lessons.append(result)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = {
            "rejected": False,
            "reason_code": None,
            "reason": "",
            "content": "Đã tạo bài học thành công",
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
                "cost_usd": _cost_from_usage("gpt-4.1", total_token_usage),
                "cached_tokens": total_token_usage["cached_tokens"],
                "cache_write_tokens": total_token_usage["cache_write_tokens"],
                "cache_savings_usd": _cache_savings_from_usage("gpt-4.1", total_token_usage),
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

    @observe(name="lesson_pipeline.stream_artifact_v3", capture_input=False, capture_output=False)
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
                "message": f"Pika đang tạo {total_lessons} bài học...",
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
                "message": f"Đã lấy thông tin bé: {child_name or 'N/A'}",
                "child_name": child_name,
                "child_age": child_age,
                "language": language,
                "image_url": pipeline_image,
            }, "pipeline")

            total_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0, "total_tokens": 0}

            # OPTIMIZATION: Generate all lessons in PARALLEL, then yield in order
            # This reduces total time from N*T to max(T1, T2, ..., TN)

            # First, yield "started" events for all lessons
            for idx, lesson_item in enumerate(request.lessons, start=1):
                yield _sse({
                    "phase": "lesson_started",
                    "index": idx,
                    "total": total_lessons,
                    "title": lesson_item.title,
                    "message": f"Đang tạo bài {idx}/{total_lessons}: {lesson_item.title}",
                    "image_url": pipeline_image,
                }, "artifact")

            # Generate all lessons in parallel
            async def generate_single(idx: int, lesson_item):
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

                lesson_id = f"lesson_{idx:03d}"
                bot_id = AgentBotId[lesson_item.agent_mode.upper()]
                template_id = lesson_item.template_id

                return {
                    "idx": idx,
                    "lesson_item": lesson_item,
                    "artifact_lesson": {
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
                    },
                    "usage": lesson_usage,
                }

            # Run all in parallel
            results = await asyncio.gather(
                *[generate_single(idx, lesson) for idx, lesson in enumerate(request.lessons, start=1)],
                return_exceptions=True,
            )

            # Sort by idx and yield in order
            artifact_lessons = []
            valid_results = [r for r in results if not isinstance(r, Exception)]
            valid_results.sort(key=lambda x: x["idx"])

            for result in valid_results:
                idx = result["idx"]
                lesson_item = result["lesson_item"]
                artifact_lesson = result["artifact_lesson"]
                lesson_usage = result["usage"]

                for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens", "total_tokens"):
                    total_token_usage[key] += lesson_usage.get(key, 0)

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
                "content": "Đã tạo bài học thành công",
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
                    "cost_usd": _cost_from_usage("gpt-4.1", total_token_usage),
                    "cached_tokens": total_token_usage["cached_tokens"],
                    "cache_write_tokens": total_token_usage["cache_write_tokens"],
                    "cache_savings_usd": _cache_savings_from_usage("gpt-4.1", total_token_usage),
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

    @observe(name="lesson_pipeline.generate_regenerate", capture_input=False, capture_output=True)
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
    @observe(name="lesson_pipeline.stream_generate", capture_input=True, capture_output=True)
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
            yield _sse({"phase": "started", "request_id": request_id, "message": "Pika đang phân tích yêu cầu...", "image_url": pipeline_image}, "pipeline")

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
                "message": f"Đã lấy thông tin bé: {user_profile.child.child_name if user_profile.child else 'N/A'}",
                "child_name": user_profile.child.child_name if user_profile.child else None,
                "child_age": user_profile.child.age if user_profile.child else None,
                "language": user_profile.language_preference,
                "image_url": pipeline_image,
            }, "pipeline")

            yield _sse({
                "phase": "memory",
                "message": "Pika đang tìm kiếm thông tin liên quan đến bé và bài học...",
                "facts": [f.text for f in memory.facts],
                "image_url": pipeline_image,
            }, "pipeline")

            if request.image_urls:
                preview = image_description[:100] + "..." if len(image_description) > 100 else image_description
                yield _sse({
                    "phase": "vision_complete",
                    "message": f"Đã trích xuất: {preview}",
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

    @observe(name="lesson_pipeline.stream_regenerate", capture_input=True, capture_output=True)
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
            yield _sse({"phase": "started", "request_id": request_id, "message": "Pika đang tạo lại bài học...", "image_url": pipeline_image}, "pipeline")

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
                "message": f"Đã lấy thông tin bé: {user_profile.child.child_name if user_profile.child else 'N/A'}",
                "child_name": user_profile.child.child_name if user_profile.child else None,
                "child_age": user_profile.child.age if user_profile.child else None,
                "language": user_profile.language_preference,
                "image_url": pipeline_image,
            }, "pipeline")

            yield _sse({
                "phase": "memory",
                "message": "Pika đang tìm kiếm thông tin liên quan đến bé và bài học...",
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
