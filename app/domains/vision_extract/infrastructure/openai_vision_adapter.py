import base64
import json
from typing import Any

import httpx
import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()


class OpenAIVisionAdapter:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-5.6-terra",  # Vision: $2/1M in, $12/1M out - quality/cost balance
        guardrail_model: str = "gpt-5.6-luna",  # Guardrail: $0.20/1M in - ultra cheap with built-in safety
        temperature: float = 0.0,
        max_tokens: int = 32768,
    ):
        self._client = client
        self._model = model
        self._guardrail_model = guardrail_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._image_cache: dict[str, str] = {}  # Cache downloaded images

    def _emit_llm(
        self,
        outcome: str,
        start_ts: float,
        *,
        error_type: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Metric-event: LLM vision OpenAI (latency/tokens/outcome). Never raises."""
        try:
            import time as _time

            from app.core.metric_events import emit as _metric_emit

            _metric_emit(
                "llm_generate",
                provider="openai",
                model=self._model,
                outcome=outcome,
                error_type=error_type,
                latency_ms=float((_time.monotonic() - start_ts) * 1000.0),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception:  # noqa: BLE001
            pass

    async def _download_image(self, url: str) -> str:
        """Download image and return base64 data URL. Uses cache to avoid re-downloading."""
        # Check cache first
        if url in self._image_cache:
            return self._image_cache[url]

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                image_data = response.content
                # Detect content type
                content_type = response.headers.get("content-type", "image/jpeg")
                b64_image = base64.b64encode(image_data).decode("utf-8")
                result = f"data:{content_type};base64,{b64_image}"
                # Cache for reuse
                self._image_cache[url] = result
                return result
        except Exception as e:
            logger.warning("image_download_failed", url=url, error=str(e))
            # Fallback to original URL
            return url

    def clear_image_cache(self) -> None:
        """Clear the image cache after request completes."""
        self._image_cache.clear()

    def _convert_to_image_url(self, url_or_base64: str) -> dict:
        """Convert URL or base64 to OpenAI image_url format."""
        if url_or_base64.startswith("data:"):
            return {"url": url_or_base64}
        else:
            return {"url": url_or_base64}

    async def extract(self, image_urls: list[str], prompt: str) -> tuple[str, dict[str, Any]]:
        """Extract content from images - returns (description, usage) tuple."""
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for url in image_urls:
            # Download image first to avoid OpenAI URL timeout issues
            image_data = await self._download_image(url)
            content.append({"type": "input_image", "image_url": image_data})

        logger.info("vision_api_call", model=self._model, image_count=len(image_urls))

        import time as _time
        _llm_start = _time.monotonic()
        try:
            # GPT-5.6 models don't support temperature parameter
            response = await self._client.responses.create(
                model=self._model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_llm("error", _llm_start, error_type=type(exc).__name__)
            raise

        raw_text = response.output_text or ""

        # Build usage dict
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            }

        logger.info("vision_api_response", tokens_used=usage.get("total_tokens", 0))
        self._emit_llm(
            "ok", _llm_start,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

        # Return tuple of (description, usage)
        return raw_text.strip(), usage

    async def check_safety(self, image_urls: list[str], prompt: str) -> tuple[bool, str]:
        """Check content safety using guardrail model. Returns (is_safe, reason). Fails open on parse error."""
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for url in image_urls:
            image_data = await self._download_image(url)  # Uses cache
            content.append({"type": "input_image", "image_url": image_data})

        logger.info("vision_safety_check", model=self._guardrail_model, image_count=len(image_urls))

        # Use Structured Outputs with strict schema for guardrail
        safety_schema = {
            "type": "object",
            "properties": {
                "safe": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["safe", "reason"],
            "additionalProperties": False
        }

        response = await self._client.responses.create(
            model=self._guardrail_model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=128,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "safety_check_output",
                    "schema": safety_schema,
                    "strict": True
                }
            },
        )

        raw_text = response.output_text or ""
        try:
            result = json.loads(raw_text.strip())
            return bool(result.get("safe", True)), str(result.get("reason", ""))
        except json.JSONDecodeError:
            logger.warning("safety_check_parse_failed", raw_text_preview=raw_text[:100])
            return True, ""  # fail open

    async def extract_v3(self, image_urls: list[str], prompt: str) -> tuple[dict, dict[str, Any]]:
        """Extract structured content from images - returns (parsed_dict, usage) tuple.

        Returns FULL lesson format matching source system:
        - rejected, reason_code, reason, content (root fields)
        - lessons array with: title, summary, detail_tasks_lesson, prompt_agent, agent_mode, options, content
        """
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for url in image_urls:
            image_data = await self._download_image(url)
            content.append({"type": "input_image", "image_url": image_data})

        logger.info("vision_api_call_v3", model=self._model, image_count=len(image_urls))

        # Use Structured Outputs with strict JSON schema - decoder enforced at sampling layer
        # This guarantees 100% schema compliance, not just valid JSON
        # UPDATED: Full lesson format matching source system
        lesson_option_schema = {
            "type": "object",
            "properties": {
                "template_id": {"type": "string"},
                "exercise_subtype": {"type": ["string", "null"]}
            },
            "required": ["template_id", "exercise_subtype"],
            "additionalProperties": False
        }

        lesson_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},  # Vietnamese summary 1-2 sentences
                "detail_tasks_lesson": {"type": "string"},  # Vietnamese, 3 activities
                "prompt_agent": {"type": "string"},  # Vietnamese, D1-D7 format
                "agent_mode": {"type": "string", "enum": ["learn_agent", "talk_agent"]},
                "options": {"type": "array", "items": lesson_option_schema},
                "content": {"type": "string"}
            },
            "required": ["title", "summary", "detail_tasks_lesson", "prompt_agent", "agent_mode", "options", "content"],
            "additionalProperties": False
        }

        output_schema = {
            "type": "object",
            "properties": {
                "rejected": {"type": "boolean"},
                "reason_code": {"type": ["string", "null"]},
                "reason": {"type": "string"},
                "content": {"type": "string"},  # Success/failure message
                "lessons": {"type": "array", "items": lesson_schema}
            },
            "required": ["rejected", "reason_code", "reason", "content", "lessons"],
            "additionalProperties": False
        }

        response = await self._client.responses.create(
            model=self._model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self._max_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "lessons_output",  # Full lesson format
                    "schema": output_schema,
                    "strict": True  # Enforced at decoder level - physically cannot violate schema
                }
            },
        )

        raw_text = response.output_text or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            }

        # Debug: log full response details
        logger.info(
            "vision_api_response_v3",
            tokens_used=usage.get("total_tokens", 0),
            output_text_length=len(raw_text),
            output_text_preview=raw_text[:200] if raw_text else "EMPTY",
            has_output=bool(response.output),
            response_status=getattr(response, 'status', 'unknown'),
        )

        try:
            result = json.loads(raw_text.strip())
            # Log the keys to debug structure issues
            logger.info("vision_v3_json_keys", keys=list(result.keys()) if isinstance(result, dict) else "not_a_dict")
        except json.JSONDecodeError:
            logger.warning("vision_v3_json_parse_failed", raw_text_preview=raw_text[:200])
            from app.core.exceptions import VisionExtractionError
            raise VisionExtractionError(
                "Vision model returned invalid JSON",
                raw_response=raw_text[:500],
            )

        return result, usage
