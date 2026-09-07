import asyncio
import base64
import hashlib
import io
import json
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

import httpx
import structlog
from langfuse import observe
from openai import AsyncOpenAI

logger = structlog.get_logger()

# Image variants sent to OpenAI. Vision models bill by resolution tiles, so
# oversized CDN images waste input tokens and upload time.
_VISION_MAX_DIM = 1536   # full detail for extraction
_GUARDRAIL_MAX_DIM = 768  # guardrail only detects unsafe content - low res is enough
_JPEG_QUALITY = 80
_CACHE_MAX_ENTRIES = 32   # bound the per-process cache (keyed by URL)

# OpenAI caches the static prefix of a prompt automatically once it exceeds this
# many tokens. Our extraction prompts are far longer, but the guardrail prompt
# sits near the boundary - log when a call is too small to ever cache.
_PROMPT_CACHE_MIN_TOKENS = 1024
_CHARS_PER_TOKEN_ESTIMATE = 4  # rough: enough to spot a prompt that can't cache


@lru_cache(maxsize=32)
def _prompt_cache_key(prompt: str) -> str:
    """Stable routing key for OpenAI's prompt cache.

    Requests sharing a cache key are routed to the same backend, which raises
    the hit rate for our handful of long static prompts. Keyed on the prompt
    text only - images vary per request and are not part of the cached prefix.
    """
    return f"lesson-vision-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:24]}"


def _warn_if_uncacheable(prompt: str, call: str) -> None:
    """Log once-per-call when a prompt is too short for the prompt cache."""
    if len(prompt) // _CHARS_PER_TOKEN_ESTIMATE < _PROMPT_CACHE_MIN_TOKENS:
        logger.info(
            "prompt_cache_unlikely",
            call=call,
            prompt_chars=len(prompt),
            min_tokens_required=_PROMPT_CACHE_MIN_TOKENS,
        )


def _resize_image_sync(raw: bytes, max_dim: int) -> tuple[bytes, str]:
    """Downscale image to max_dim on the long edge. Returns (bytes, mime).

    Falls back to original bytes if Pillow can't decode (e.g. exotic format).
    Runs in a thread via asyncio.to_thread - PIL is CPU-bound.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
        if max(img.size) <= max_dim:
            return raw, Image.MIME.get(img.format or "", "image/jpeg")
        img.thumbnail((max_dim, max_dim))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=_JPEG_QUALITY)
        return out.getvalue(), "image/jpeg"
    except Exception as exc:  # noqa: BLE001
        logger.warning("image_resize_failed", error=str(exc))
        return raw, "image/jpeg"


class _LessonStreamScanner:
    """Incrementally extracts complete lesson objects from a streamed JSON text.

    The model streams the output_schema JSON: {..., "lessons": [ {...}, {...} ]}.
    We locate the lessons array once, then emit each top-level object inside it
    as soon as its braces balance (string/escape aware).
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_array = False
        self._pos = 0           # scan cursor
        self._depth = 0         # brace depth inside current lesson object
        self._obj_start = -1    # index where current lesson object begins
        self._in_string = False
        self._escape = False

    def feed(self, delta: str) -> list[dict]:
        self._buf += delta
        if not self._in_array:
            marker = self._buf.find('"lessons"')
            if marker == -1:
                return []
            bracket = self._buf.find("[", marker)
            if bracket == -1:
                return []
            self._in_array = True
            self._pos = bracket + 1

        completed: list[dict] = []
        while self._pos < len(self._buf):
            ch = self._buf[self._pos]
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
            elif ch == '"':
                self._in_string = True
            elif ch == "{":
                if self._depth == 0:
                    self._obj_start = self._pos
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0 and self._obj_start != -1:
                    chunk = self._buf[self._obj_start : self._pos + 1]
                    try:
                        # strict=False: tolerate literal newlines in strings
                        completed.append(json.loads(chunk, strict=False))
                    except json.JSONDecodeError:
                        logger.warning("lesson_stream_parse_failed", chunk_preview=chunk[:120])
                    self._obj_start = -1
            elif ch == "]" and self._depth == 0:
                # lessons array closed - nothing more to scan for
                self._pos = len(self._buf)
                break
            self._pos += 1
        return completed


def _build_suggested_lessons_schema() -> dict:
    """Schema for v3/lessons/generate - suggestions only, no full prompt.

    This matches the source system's v3 flow where generate returns suggestions
    and generate_artifact creates the full lesson with checkpoints.
    """
    option_schema = {
        "type": "object",
        "properties": {
            "template_id": {"type": "string"},
            "exercise_subtype": {"type": ["string", "null"]},
            "option": {"type": "string"},  # Human-readable action name
        },
        "required": ["template_id", "exercise_subtype", "option"],
        "additionalProperties": False,
    }
    suggestion_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "agent_mode": {"type": "string", "enum": ["learn_agent", "talk_agent"]},
            "content": {"type": "string"},  # Raw content extracted from image
            "options": {"type": "array", "items": option_schema},
        },
        "required": ["title", "agent_mode", "content", "options"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "rejected": {"type": "boolean"},
            "reason_code": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "suggested_lessons": {"type": "array", "items": suggestion_schema},
        },
        "required": ["rejected", "reason_code", "reason", "suggested_lessons"],
        "additionalProperties": False,
    }


def _build_lessons_output_schema() -> dict:
    """Strict JSON schema for the full lesson format (used by v1 flow)."""
    lesson_option_schema = {
        "type": "object",
        "properties": {
            "template_id": {"type": "string"},
            "exercise_subtype": {"type": ["string", "null"]},
        },
        "required": ["template_id", "exercise_subtype"],
        "additionalProperties": False,
    }
    lesson_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},                # Vietnamese, 1-2 sentences
            "detail_tasks_lesson": {"type": "string"},    # Vietnamese, 3 activities
            "prompt_agent": {"type": "string"},           # Vietnamese, D1-D7 format
            "agent_mode": {"type": "string", "enum": ["learn_agent", "talk_agent"]},
            "options": {"type": "array", "items": lesson_option_schema},
            "content": {"type": "string"},
        },
        "required": ["title", "summary", "detail_tasks_lesson", "prompt_agent", "agent_mode", "options", "content"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "rejected": {"type": "boolean"},
            "reason_code": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "content": {"type": "string"},
            "lessons": {"type": "array", "items": lesson_schema},
        },
        "required": ["rejected", "reason_code", "reason", "content", "lessons"],
        "additionalProperties": False,
    }


_SUGGESTED_LESSONS_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "suggested_lessons_output",
        "schema": _build_suggested_lessons_schema(),
        "strict": True,
    }
}

_LESSONS_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "lessons_output",
        "schema": _build_lessons_output_schema(),
        "strict": True,  # decoder-enforced - physically cannot violate schema
    }
}


class OpenAIVisionAdapter:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-5.6-terra",  # Vision: $2/1M in, $12/1M out - quality/cost balance
        guardrail_model: str = "gpt-5.6-luna",  # Guardrail: $0.20/1M in - ultra cheap with built-in safety
        suggestions_model: str | None = None,  # v3/lessons/generate; defaults to `model`
        v1_extraction_model: str | None = None,  # v1 extract(): plain describe - faster model OK
        temperature: float = 0.0,
        max_tokens: int = 32768,
    ):
        self._client = client
        self._model = model
        self._guardrail_model = guardrail_model
        # v3 suggestions only need page segmentation + template choice, so this
        # can be a smaller/faster model than full extraction.
        self._suggestions_model = suggestions_model or model
        # v1 extraction is a simple describe/OCR step feeding the 5-expert
        # generation prompt - GPT-4.1 streams output ~2.5x faster than Terra.
        self._v1_extraction_model = v1_extraction_model or model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Shared HTTP client: one connection pool, no per-download TLS handshake
        self._http: httpx.AsyncClient | None = None
        # url -> in-flight/finished download task (raw bytes | None on failure).
        # Concurrent guardrail + vision calls share ONE download per URL.
        self._download_tasks: dict[str, asyncio.Task] = {}
        # (url, variant) -> data URL
        self._variant_cache: dict[tuple[str, str], str] = {}

    @property
    def model(self) -> str:
        """Model used for full vision extraction (v1 flow)."""
        return self._model

    @property
    def suggestions_model(self) -> str:
        """Model used for v3/lessons/generate suggestions."""
        return self._suggestions_model

    @property
    def guardrail_model(self) -> str:
        """Model used for the safety/educational-value guardrail."""
        return self._guardrail_model

    def _emit_llm(
        self,
        outcome: str,
        start_ts: float,
        *,
        error_type: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cached_tokens: int | None = None,
        model: str | None = None,
    ) -> None:
        """Metric-event: LLM vision OpenAI (latency/tokens/outcome). Never raises."""
        try:
            import time as _time

            from app.core.metric_events import emit as _metric_emit

            cache_hit_ratio = None
            if prompt_tokens and cached_tokens is not None:
                cache_hit_ratio = round(cached_tokens / prompt_tokens, 3)

            _metric_emit(
                "llm_generate",
                provider="openai",
                model=model or self._model,
                outcome=outcome,
                error_type=error_type,
                latency_ms=float((_time.monotonic() - start_ts) * 1000.0),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_hit_ratio=cache_hit_ratio,
            )
        except Exception:  # noqa: BLE001
            pass

    def _log_cache(self, call: str, usage: dict, model: str | None = None) -> None:
        """Structured log of prompt-cache effectiveness for this call."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        cached = usage.get("cached_tokens", 0)
        logger.info(
            "prompt_cache_usage",
            call=call,
            model=model or self._model,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            cache_write_tokens=usage.get("cache_write_tokens", 0),
            cache_hit_ratio=round(cached / prompt_tokens, 3) if prompt_tokens else 0.0,
        )

    # ------------------------------------------------------------------
    # Image download & preparation
    # ------------------------------------------------------------------
    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=10))
        return self._http

    async def _download_raw(self, url: str) -> bytes | None:
        """Fetch image bytes, retrying once on a transient network failure."""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._get_http().get(url, follow_redirects=True)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as e:
                # A 4xx is the CDN's answer, not a blip - don't retry.
                logger.warning("image_download_failed", url=url, error=str(e))
                return None
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt == 0:
                    await asyncio.sleep(0.5)
        logger.warning("image_download_failed", url=url, error=str(last_error))
        return None

    def _evict_if_full(self) -> None:
        if len(self._download_tasks) > _CACHE_MAX_ENTRIES:
            self._download_tasks.clear()
            self._variant_cache.clear()

    async def _get_image(self, url: str, variant: str) -> str:
        """Return data URL for the requested variant ('full' | 'low').

        Raw bytes are downloaded at most once per URL (shared task), then each
        variant is resized+encoded once and cached. Falls back to the original
        URL when download fails (OpenAI fetches it directly).
        """
        key = (url, variant)
        if key in self._variant_cache:
            return self._variant_cache[key]

        task = self._download_tasks.get(url)
        if task is None:
            self._evict_if_full()
            task = asyncio.ensure_future(self._download_raw(url))
            self._download_tasks[url] = task
        raw = await task

        if raw is None:
            # Drop the failed task so the next request retries instead of
            # replaying the failure - a single CDN blip used to poison this
            # URL until the cache filled up, sending every later request down
            # the URL fallback (which OpenAI then times out on too).
            if self._download_tasks.get(url) is task:
                del self._download_tasks[url]
            return url  # fallback: let OpenAI fetch the URL itself

        max_dim = _VISION_MAX_DIM if variant == "full" else _GUARDRAIL_MAX_DIM
        resized, mime = await asyncio.to_thread(_resize_image_sync, raw, max_dim)
        data_url = f"data:{mime};base64,{base64.b64encode(resized).decode('utf-8')}"
        self._variant_cache[key] = data_url
        return data_url

    async def _build_image_content(self, image_urls: list[str], variant: str) -> list[dict]:
        """Download + prepare all images IN PARALLEL and return input_image items."""
        images = await asyncio.gather(*(self._get_image(url, variant) for url in image_urls))
        detail = "low" if variant == "low" else "auto"
        return [{"type": "input_image", "image_url": img, "detail": detail} for img in images]

    def clear_image_cache(self) -> None:
        """Clear cached downloads (kept for API compatibility)."""
        self._download_tasks.clear()
        self._variant_cache.clear()

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------
    @observe(name="vision_llm_call", capture_input=True, capture_output=True)
    async def extract(self, image_urls: list[str], prompt: str) -> tuple[str, dict[str, Any]]:
        """Extract content from images - returns (description, usage) tuple."""
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        content += await self._build_image_content(image_urls, "full")

        model = self._v1_extraction_model
        logger.info("vision_api_call", model=model, image_count=len(image_urls))
        _warn_if_uncacheable(prompt, "extract")

        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self._max_tokens,
            "prompt_cache_key": _prompt_cache_key(prompt),
        }
        # reasoning "none": extraction is deterministic OCR/describe work -
        # hidden reasoning tokens only add latency. Only GPT-5.6 models accept
        # the reasoning parameter (GPT-4.x rejects it).
        if model.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "none"}

        import time as _time
        _llm_start = _time.monotonic()
        try:
            # GPT-5.6 models don't support temperature parameter
            response = await self._client.responses.create(**request_kwargs)
        except Exception as exc:  # noqa: BLE001
            self._emit_llm("error", _llm_start, error_type=type(exc).__name__)
            raise

        raw_text = response.output_text or ""
        usage = self._usage_dict(response)

        logger.info("vision_api_response", tokens_used=usage.get("total_tokens", 0))
        self._log_cache("extract", usage)
        self._emit_llm(
            "ok", _llm_start,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cached_tokens=usage.get("cached_tokens"),
        )

        return raw_text.strip(), usage

    @observe(name="vision_safety_check", capture_input=False, capture_output=False)
    async def check_safety(self, image_urls: list[str], prompt: str) -> tuple[bool, str]:
        """Check content safety using guardrail model. Returns (is_safe, reason). Fails open on parse error.

        Uses LOW-RES image variant + detail:low - detecting unsafe content does
        not need full resolution, and this cuts guardrail input tokens sharply.
        """
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        content += await self._build_image_content(image_urls, "low")

        logger.info("vision_safety_check", model=self._guardrail_model, image_count=len(image_urls))

        safety_schema = {
            "type": "object",
            "properties": {
                "safe": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["safe", "reason"],
            "additionalProperties": False,
        }

        response = await self._client.responses.create(
            model=self._guardrail_model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=128,
            prompt_cache_key=_prompt_cache_key(prompt),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "safety_check_output",
                    "schema": safety_schema,
                    "strict": True,
                }
            },
        )

        raw_text = response.output_text or ""
        self._log_cache("check_safety", self._usage_dict(response), model=self._guardrail_model)
        try:
            result = json.loads(raw_text.strip())
            return bool(result.get("safe", True)), str(result.get("reason", ""))
        except json.JSONDecodeError:
            logger.warning("safety_check_parse_failed", raw_text_preview=raw_text[:100])
            return True, ""  # fail open

    def _check_truncation(self, response: Any, raw_text: str, usage: dict) -> None:
        """Raise a clear error when the response hit max_output_tokens."""
        if getattr(response, "status", None) == "incomplete":
            incomplete = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete, "reason", "unknown") if incomplete else "unknown"
            logger.error(
                "vision_v3_response_incomplete",
                reason=reason,
                max_output_tokens=self._max_tokens,
                completion_tokens=usage.get("completion_tokens", 0),
            )
            from app.core.exceptions import VisionExtractionError
            raise VisionExtractionError(
                f"Vision response truncated ({reason}). "
                f"Content requires more than max_output_tokens={self._max_tokens} - "
                f"reduce image count or raise OPENAI_VISION_MAX_TOKENS.",
                raw_response=raw_text[:500],
            )

    @staticmethod
    def _usage_dict(response: Any) -> dict:
        if not getattr(response, "usage", None):
            return {}
        prompt_tokens = response.usage.input_tokens or 0
        completion_tokens = response.usage.output_tokens or 0
        # OpenAI reports the cached subset of the prompt under
        # input_tokens_details. cached_tokens = served FROM cache (discounted);
        # cache_write_tokens = a miss that populated the cache for next time.
        details = getattr(response.usage, "input_tokens_details", None)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
            "cache_write_tokens": getattr(details, "cache_write_tokens", 0) or 0,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @observe(name="vision_llm_call_v3", capture_input=True, capture_output=True)
    async def extract_v3(
        self,
        image_urls: list[str],
        prompt: str,
        *,
        reasoning_effort: str | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """Extract structured content from images - returns (parsed_dict, usage) tuple.

        Returns FULL lesson format matching source system:
        - rejected, reason_code, reason, content (root fields)
        - lessons array with: title, summary, detail_tasks_lesson, prompt_agent, agent_mode, options, content

        Args:
            reasoning_effort: Optional reasoning effort level ("minimal", "low", "medium", "high").
                             Use "minimal" for fast extraction tasks to reduce latency.
        """
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        content += await self._build_image_content(image_urls, "full")

        logger.info("vision_api_call_v3", model=self._model, image_count=len(image_urls), reasoning_effort=reasoning_effort)
        _warn_if_uncacheable(prompt, "extract_v3")

        # Build request kwargs - add reasoning_effort if specified
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self._max_tokens,
            "prompt_cache_key": _prompt_cache_key(prompt),
            "text": _LESSONS_TEXT_FORMAT,
        }
        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        response = await self._client.responses.create(**request_kwargs)

        raw_text = response.output_text or ""
        usage = self._usage_dict(response)
        self._log_cache("extract_v3", usage)

        logger.info(
            "vision_api_response_v3",
            tokens_used=usage.get("total_tokens", 0),
            output_text_length=len(raw_text),
            output_text_preview=raw_text[:200] if raw_text else "EMPTY",
            has_output=bool(response.output),
            response_status=getattr(response, "status", "unknown"),
        )

        self._check_truncation(response, raw_text, usage)

        try:
            result = json.loads(raw_text.strip())
            logger.info("vision_v3_json_keys", keys=list(result.keys()) if isinstance(result, dict) else "not_a_dict")
        except json.JSONDecodeError:
            logger.warning("vision_v3_json_parse_failed", raw_text_preview=raw_text[:200])
            from app.core.exceptions import VisionExtractionError
            raise VisionExtractionError(
                "Vision model returned invalid JSON",
                raw_response=raw_text[:500],
            )

        return result, usage

    @observe(name="vision_llm_call_v3_cached", capture_input=True, capture_output=True)
    async def extract_v3_cached(
        self,
        image_urls: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict, dict[str, Any]]:
        """Extract with SYSTEM + USER prompts for optimal prompt caching.

        The system prompt is static and cached by OpenAI when it exceeds ~1024 tokens.
        This can reduce TTFT by up to 80% and input costs by up to 90%.

        Args:
            image_urls: List of image URLs to analyze
            system_prompt: Static system instructions (should be >1024 tokens for caching)
            user_prompt: Dynamic per-request content (subject, memory, etc.)
        """
        # Build user content with images
        user_content: list[dict] = [{"type": "input_text", "text": user_prompt}]
        user_content += await self._build_image_content(image_urls, "full")

        logger.info(
            "vision_api_call_v3_cached",
            model=self._model,
            image_count=len(image_urls),
            system_prompt_chars=len(system_prompt),
            user_prompt_chars=len(user_prompt),
        )

        # System prompt should be cacheable (>1024 tokens)
        system_tokens_est = len(system_prompt) // 4
        if system_tokens_est < 1024:
            logger.warning(
                "prompt_cache_may_not_trigger",
                system_tokens_est=system_tokens_est,
                min_required=1024,
            )

        response = await self._client.responses.create(
            model=self._model,
            instructions=system_prompt,  # System prompt - cached by OpenAI
            input=[{"role": "user", "content": user_content}],
            max_output_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(system_prompt),  # Stable cache key
            text=_LESSONS_TEXT_FORMAT,
        )

        raw_text = response.output_text or ""
        usage = self._usage_dict(response)
        self._log_cache("extract_v3_cached", usage)

        logger.info(
            "vision_api_response_v3_cached",
            tokens_used=usage.get("total_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            output_text_length=len(raw_text),
            has_output=bool(response.output),
        )

        self._check_truncation(response, raw_text, usage)

        try:
            result = json.loads(raw_text.strip())
            logger.info("vision_v3_cached_json_keys", keys=list(result.keys()) if isinstance(result, dict) else "not_a_dict")
        except json.JSONDecodeError:
            logger.warning("vision_v3_cached_json_parse_failed", raw_text_preview=raw_text[:200])
            from app.core.exceptions import VisionExtractionError
            raise VisionExtractionError(
                "Vision model returned invalid JSON",
                raw_response=raw_text[:500],
            )

        return result, usage

    @observe(name="vision_llm_call_v3_suggestions", capture_input=False, capture_output=False)
    async def extract_v3_suggestions(self, image_urls: list[str], prompt: str) -> tuple[dict, dict[str, Any]]:
        """Extract suggested lessons from images - clone of source v3/lessons/generate.

        Returns format matching source:
        - suggested_lessons array with: title, agent_mode, content, options

        Uses json_object format like source (no strict schema).
        """
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        content += await self._build_image_content(image_urls, "full")

        # v3 runs on the (smaller) suggestions model, not the full vision model.
        model = self._suggestions_model
        logger.info("vision_api_call_v3_suggestions", model=model, image_count=len(image_urls))

        # Clone source: use json_object format, not strict schema
        # Note: GPT-5.6 models don't support temperature parameter
        response = await self._client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(prompt),
            text={"format": {"type": "json_object"}},
        )

        raw_text = response.output_text or ""
        usage = self._usage_dict(response)
        self._log_cache("extract_v3_suggestions", usage, model=model)

        logger.info(
            "vision_api_response_v3_suggestions",
            tokens_used=usage.get("total_tokens", 0),
            output_text_length=len(raw_text),
            output_text_preview=raw_text[:200] if raw_text else "EMPTY",
            has_output=bool(response.output),
            response_status=getattr(response, "status", "unknown"),
        )

        self._check_truncation(response, raw_text, usage)

        try:
            result = json.loads(raw_text.strip())
            logger.info("vision_v3_suggestions_keys", keys=list(result.keys()) if isinstance(result, dict) else "not_a_dict")
        except json.JSONDecodeError:
            logger.warning("vision_v3_suggestions_parse_failed", raw_text_preview=raw_text[:200])
            from app.core.exceptions import VisionExtractionError
            raise VisionExtractionError(
                "Vision model returned invalid JSON",
                raw_response=raw_text[:500],
            )

        return result, usage

    @observe(name="vision_llm_call_v3_stream", capture_input=False, capture_output=False)
    async def extract_v3_stream(
        self, image_urls: list[str], prompt: str, model: str | None = None
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Streaming variant of extract_v3.

        Args:
            model: overrides the full-extraction model. The v3 flow passes the
                   (smaller) suggestions model; v1 leaves it unset.

        Yields:
            ("lesson", lesson_dict)          - as soon as each lesson's JSON completes
            ("complete", (result, usage))    - final parsed result + token usage
        """
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        content += await self._build_image_content(image_urls, "full")

        model = model or self._model
        logger.info("vision_api_call_v3_stream", model=model, image_count=len(image_urls))
        _warn_if_uncacheable(prompt, "extract_v3_stream")

        stream = await self._client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(prompt),
            text=_LESSONS_TEXT_FORMAT,
            stream=True,
        )

        scanner = _LessonStreamScanner()
        raw_text = ""
        final_response: Any = None

        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type.endswith("output_text.delta"):
                delta = getattr(event, "delta", "") or ""
                raw_text += delta
                for lesson in scanner.feed(delta):
                    yield ("lesson", lesson)
            elif event_type in ("response.completed", "response.incomplete", "response.failed"):
                final_response = getattr(event, "response", None)

        usage = self._usage_dict(final_response) if final_response is not None else {}
        self._log_cache("extract_v3_stream", usage, model=model)

        logger.info(
            "vision_api_response_v3_stream",
            tokens_used=usage.get("total_tokens", 0),
            output_text_length=len(raw_text),
            response_status=getattr(final_response, "status", "unknown"),
        )

        if final_response is not None:
            self._check_truncation(final_response, raw_text, usage)

        try:
            result = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            logger.warning("vision_v3_stream_json_parse_failed", raw_text_preview=raw_text[:200])
            from app.core.exceptions import VisionExtractionError
            raise VisionExtractionError(
                "Vision model returned invalid JSON",
                raw_response=raw_text[:500],
            )

        yield ("complete", (result, usage))
