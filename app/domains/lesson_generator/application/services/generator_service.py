"""GeneratorService: builds prompt, calls GPT-4.1, parses expert discussion + JSON."""

import json
import re
import time
from collections.abc import AsyncGenerator

import structlog
from langfuse import observe

from app.domains.lesson_generator.application.services.prompt_builder import (
    _TALK_AGENT_S0_S1,
    build_artifact_prompt,
    build_lesson_prompt,
    build_regenerate_lesson_prompt,
    build_talk_agent_section_2,
    build_talk_agent_sections_3_to_5_from_json,
    build_talk_agent_sections_6_to_8,
    build_talk_agent_system_task_prompt,
)
from app.domains.lesson_generator.infrastructure.openai_lesson_adapter import OpenAILessonAdapter

logger = structlog.get_logger()


_EXPERT_PATTERN = re.compile(
    r"\*?\*?\[([A-E])\]\s*([^:*]+?)\*?\*?:\s*",
)

_EXPERT_MAP = {
    "A": "vision_analyst",
    "B": "curriculum_designer",
    "C": "child_psychologist",
    "D": "safety_reviewer",
    "E": "final_editor",
}


def _extract_json_from_text(text: str) -> dict:
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    last_brace = text.rfind("}")
    if last_brace == -1:
        raise ValueError("No JSON found in LLM response")

    depth = 0
    start = -1
    for i in range(last_brace, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break

    if start == -1:
        raise ValueError("No balanced JSON object found")

    return json.loads(text[start : last_brace + 1])


def _extract_expert_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    matches = list(_EXPERT_PATTERN.finditer(text))

    for i, m in enumerate(matches):
        letter = m.group(1)
        expert_key = _EXPERT_MAP.get(letter, letter.lower())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if expert_key == "final_editor":
            json_start = content.find("```json")
            if json_start != -1:
                content = content[:json_start].strip()
            elif content.find("{") != -1:
                content = content[: content.find("{")].strip()

        if content:
            sections.append((expert_key, f"[{letter}] {m.group(2).strip()}: {content}"))

    return sections


def _extract_user_message(prompt: str) -> str:
    """Extract user message from prompt for logging."""
    if len(prompt) > 200:
        return prompt[:200] + "..."
    return prompt


class GeneratorService:
    def __init__(self, adapter: OpenAILessonAdapter, model_name: str = "gpt-4.1"):
        self._adapter = adapter
        self._model_name = model_name

    @observe(name="lesson_generation", capture_input=True, capture_output=True)
    async def generate_lesson(
        self,
        *,
        extracted_content: dict,
        subject: str,
        purpose: str,
        language: str,
        memory_facts: list | None = None,
        parent_notes: str | None = None,
        child_age: int | None = None,
        child_name: str | None = None,
    ) -> tuple[str, dict]:
        start = time.monotonic()

        prompt = build_lesson_prompt(
            extracted_content=extracted_content,
            subject=subject,
            purpose=purpose,
            language=language,
            memory_facts=memory_facts,
            parent_notes=parent_notes,
            child_age=child_age,
            child_name=child_name,
        )

        # Log LLM request
        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            input_content=_extract_user_message(prompt),
        )

        raw_response, usage = await self._adapter.generate(prompt)
        expert_sections = _extract_expert_sections(raw_response)
        expert_log = "\n".join(content for _, content in expert_sections)
        lesson_plan = _extract_json_from_text(raw_response)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Log LLM response with token usage
        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            topic=lesson_plan.get("topic", "unknown"),
            activities_count=len(lesson_plan.get("activities", [])),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        return expert_log, lesson_plan

    @observe(name="lesson_regeneration", capture_input=True, capture_output=True)
    async def generate_regenerate_lesson(
        self,
        *,
        original_lesson,
        extracted_content: dict,
        subject: str,
        purpose: str,
        language: str,
        memory_facts: list | None = None,
        parent_notes: str | None = None,
        child_age: int | None = None,
        child_name: str | None = None,
    ) -> tuple[str, dict]:
        start = time.monotonic()

        prompt = build_regenerate_lesson_prompt(
            original_lesson=original_lesson,
            extracted_content=extracted_content,
            subject=subject,
            purpose=purpose,
            language=language,
            memory_facts=memory_facts,
            parent_notes=parent_notes,
            child_age=child_age,
            child_name=child_name,
        )

        # Log LLM request
        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            input_content=_extract_user_message(prompt),
        )

        raw_response, usage = await self._adapter.generate(prompt)
        expert_sections = _extract_expert_sections(raw_response)
        expert_log = "\n".join(content for _, content in expert_sections)
        lesson_plan = _extract_json_from_text(raw_response)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Log LLM response with token usage
        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            topic=lesson_plan.get("topic", "unknown"),
            activities_count=len(lesson_plan.get("activities", [])),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        return expert_log, lesson_plan

    async def generate_artifact_for_lesson(
        self,
        *,
        lesson_title: str,
        lesson_content: str,
        lesson_option: str,
        agent_mode: str,
        language: str = "vi",
        child_name: str | None = None,
        child_age: int | None = None,
        custom_prompt: str | None = None,
        template_id: str | None = None,
    ) -> tuple[dict, dict]:
        """Generate artifact JSON for a single lesson item.

        Returns:
            (artifact_dict, usage_dict)
        """
        start = time.monotonic()

        prompt = build_artifact_prompt(
            lesson_title=lesson_title,
            lesson_content=lesson_content,
            lesson_option=lesson_option,
            agent_mode=agent_mode,
            language=language,
            child_name=child_name,
            child_age=child_age,
            custom_prompt=custom_prompt,
            template_id=template_id,
        )

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="ARTIFACT",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            lesson_title=lesson_title,
            template_id=template_id,
            agent_mode="learn_agent",
        )

        raw_response, usage = await self._adapter.generate(prompt)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="ARTIFACT",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            lesson_title=lesson_title,
            template_id=template_id,
            agent_mode="learn_agent",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        try:
            artifact = _extract_json_from_text(raw_response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(
                "learn_agent.json_parse_error",
                log_type="external_api",
                feature="ARTIFACT",
                target_service="openai",
                lesson_title=lesson_title,
                template_id=template_id,
                agent_mode="learn_agent",
                error=str(e),
                raw_response_preview=raw_response[:200] if raw_response else "",
                duration_ms=elapsed_ms,
            )
            raise

        return artifact, usage

    async def generate_talk_agent_system_task(
        self,
        *,
        lesson_content: str,
        template_id: str,
        language: str = "vi",
        child_name: str | None = None,
        child_age: int | None = None,
        memory_facts: list | None = None,
        custom_prompt: str | None = None,
    ) -> tuple[dict, dict]:
        """Generate system_task_description, summary, and detail_tasks_lesson for a talk_agent lesson.

        Returns:
            (artifact_data, usage_dict) where artifact_data has keys:
            system_task_description, summary, detail_tasks_lesson
        """
        start = time.monotonic()

        prompt = build_talk_agent_system_task_prompt(
            lesson_content=lesson_content,
            template_id=template_id,
            child_name=child_name,
            child_age=child_age,
            custom_prompt=custom_prompt,
        )

        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="ARTIFACT",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            template_id=template_id,
            agent_mode="talk_agent",
        )

        raw_response, usage = await self._adapter.generate(prompt)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "external.api.success",
            log_type="external_api",
            feature="ARTIFACT",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            status_code=200,
            duration_ms=elapsed_ms,
            template_id=template_id,
            agent_mode="talk_agent",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        try:
            parsed = _extract_json_from_text(raw_response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(
                "talk_agent.json_parse_error",
                log_type="external_api",
                feature="ARTIFACT",
                target_service="openai",
                template_id=template_id,
                agent_mode="talk_agent",
                error=str(e),
                raw_response_preview=raw_response[:200] if raw_response else "",
                duration_ms=elapsed_ms,
            )
            raise

        sections_3_to_5 = build_talk_agent_sections_3_to_5_from_json(parsed)
        sections_6_to_8 = build_talk_agent_sections_6_to_8(
            child_name=child_name,
            child_age=child_age,
            memory_facts=memory_facts,
        )
        system_task_description = (
            _TALK_AGENT_S0_S1
            + build_talk_agent_section_2(language)
            + sections_3_to_5
            + sections_6_to_8
        )
        summary = parsed.get("conversation_goal", "")
        detail_tasks_lesson = parsed.get("detail_tasks_lesson", "")
        return {
            "system_task_description": system_task_description,
            "summary": summary,
            "detail_tasks_lesson": detail_tasks_lesson,
        }, usage

    async def stream_generate_lesson(
        self,
        *,
        extracted_content: dict,
        subject: str,
        purpose: str,
        language: str,
        memory_facts: list | None = None,
        parent_notes: str | None = None,
        child_age: int | None = None,
        child_name: str | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Yields (event_type, content) tuples as the LLM streams.

        event_type is one of: "thinking:{expert_key}", "json_chunk", "done"
        """
        prompt = build_lesson_prompt(
            extracted_content=extracted_content,
            subject=subject,
            purpose=purpose,
            language=language,
            memory_facts=memory_facts,
            parent_notes=parent_notes,
            child_age=child_age,
            child_name=child_name,
        )

        # Log LLM request start
        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            input_content=_extract_user_message(prompt),
        )

        buffer = ""
        current_expert: str | None = None
        current_expert_content = ""
        json_accumulator = ""
        in_json_block = False

        async for chunk, usage in self._adapter.stream_generate(prompt):
            buffer += chunk

            if in_json_block:
                json_accumulator += chunk
                continue

            if "```json" in buffer and not in_json_block:
                in_json_block = True
                json_start = buffer.find("```json")
                pre_json = buffer[:json_start]

                if current_expert and current_expert_content:
                    current_expert_content += pre_json.split("**")[-1] if "**" in pre_json else pre_json
                    yield (f"thinking:{current_expert}", current_expert_content.strip())
                    current_expert_content = ""

                json_accumulator = buffer[json_start + len("```json"):]
                buffer = ""
                continue

            for m in _EXPERT_PATTERN.finditer(buffer):
                letter = m.group(1)
                new_expert = _EXPERT_MAP.get(letter, letter.lower())

                if current_expert and current_expert and current_expert_content.strip():
                    section_end = buffer[:m.start()]
                    current_expert_content += section_end
                    yield (f"thinking:{current_expert}", current_expert_content.strip())
                    current_expert_content = ""

                current_expert = new_expert
                buffer = buffer[m.end():]
                break
            else:
                if current_expert:
                    if len(buffer) > 200:
                        current_expert_content += buffer[:-50]
                        buffer = buffer[-50:]

        if current_expert and current_expert_content.strip():
            yield (f"thinking:{current_expert}", current_expert_content.strip())

        if buffer.strip():
            if current_expert:
                yield (f"thinking:{current_expert}", buffer.strip())

        full_json_text = json_accumulator
        if "```" in full_json_text:
            full_json_text = full_json_text[: full_json_text.find("```")]

        try:
            lesson_plan = json.loads(full_json_text.strip())
            yield ("json_complete", json.dumps(lesson_plan, ensure_ascii=False))

            # Log success after streaming completes
            logger.info(
                "external.api.success",
                log_type="external_api",
                feature="LLM",
                target_service="openai",
                target_endpoint="https://api.openai.com/v1/chat/completions",
                http_method="POST",
                model=self._model_name,
                status_code=200,
                topic=lesson_plan.get("topic", "unknown"),
                activities_count=len(lesson_plan.get("activities", [])),
            )

        except json.JSONDecodeError:
            full_text = buffer + json_accumulator
            try:
                lesson_plan = _extract_json_from_text(full_text)
                yield ("json_complete", json.dumps(lesson_plan, ensure_ascii=False))
            except (ValueError, json.JSONDecodeError) as exc:
                logger.error(
                    "external.api.error",
                    log_type="external_api",
                    feature="LLM",
                    target_service="openai",
                    target_endpoint="https://api.openai.com/v1/chat/completions",
                    http_method="POST",
                    model=self._model_name,
                    error_type="JSONDecodeError",
                    error_message=str(exc),
                )
                yield ("error", f"Failed to parse lesson JSON: {exc}")

    async def stream_regenerate_lesson(
        self,
        *,
        original_lesson,
        extracted_content: dict,
        subject: str,
        purpose: str,
        language: str,
        memory_facts: list | None = None,
        parent_notes: str | None = None,
        child_age: int | None = None,
        child_name: str | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Yields (event_type, content) tuples for regeneration."""
        prompt = build_regenerate_lesson_prompt(
            original_lesson=original_lesson,
            extracted_content=extracted_content,
            subject=subject,
            purpose=purpose,
            language=language,
            memory_facts=memory_facts,
            parent_notes=parent_notes,
            child_age=child_age,
            child_name=child_name,
        )

        # Log LLM request start
        logger.info(
            "external.api.start",
            log_type="external_api",
            feature="LLM",
            target_service="openai",
            target_endpoint="https://api.openai.com/v1/chat/completions",
            http_method="POST",
            model=self._model_name,
            input_content=_extract_user_message(prompt),
        )

        buffer = ""
        current_expert: str | None = None
        current_expert_content = ""
        json_accumulator = ""
        in_json_block = False

        async for chunk, usage in self._adapter.stream_generate(prompt):
            buffer += chunk

            if in_json_block:
                json_accumulator += chunk
                continue

            if "```json" in buffer and not in_json_block:
                in_json_block = True
                json_start = buffer.find("```json")
                pre_json = buffer[:json_start]

                if current_expert and current_expert_content:
                    current_expert_content += pre_json.split("**")[-1] if "**" in pre_json else pre_json
                    yield (f"thinking:{current_expert}", current_expert_content.strip())
                    current_expert_content = ""

                json_accumulator = buffer[json_start + len("```json"):]
                buffer = ""
                continue

            for m in _EXPERT_PATTERN.finditer(buffer):
                letter = m.group(1)
                new_expert = _EXPERT_MAP.get(letter, letter.lower())

                if current_expert and current_expert and current_expert_content.strip():
                    section_end = buffer[:m.start()]
                    current_expert_content += section_end
                    yield (f"thinking:{current_expert}", current_expert_content.strip())
                    current_expert_content = ""

                current_expert = new_expert
                buffer = buffer[m.end():]
                break
            else:
                if current_expert:
                    if len(buffer) > 200:
                        current_expert_content += buffer[:-50]
                        buffer = buffer[-50:]

        if current_expert and current_expert_content.strip():
            yield (f"thinking:{current_expert}", current_expert_content.strip())

        if buffer.strip():
            if current_expert:
                yield (f"thinking:{current_expert}", buffer.strip())

        full_json_text = json_accumulator
        if "```" in full_json_text:
            full_json_text = full_json_text[: full_json_text.find("```")]

        try:
            lesson_plan = json.loads(full_json_text.strip())
            yield ("json_complete", json.dumps(lesson_plan, ensure_ascii=False))

            # Log success
            logger.info("external.api.success", feature="LLM", status_code=200)

        except json.JSONDecodeError:
            full_text = buffer + json_accumulator
            try:
                lesson_plan = _extract_json_from_text(full_text)
                yield ("json_complete", json.dumps(lesson_plan, ensure_ascii=False))
            except (ValueError, json.JSONDecodeError) as exc:
                yield ("error", f"Failed to parse lesson JSON: {exc}")
