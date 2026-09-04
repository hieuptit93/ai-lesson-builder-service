import hashlib
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

import structlog
from langfuse import observe
from openai import AsyncOpenAI

logger = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are an expert team of educators creating personalized lessons for "
    "children aged 4-8. Always follow the expert discussion format and output "
    "valid JSON at the end."
)


@lru_cache(maxsize=8)
def _prompt_cache_key(system_prompt: str) -> str:
    """Stable routing key so repeat calls hit the same prompt cache shard."""
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:24]
    return f"lesson-gen-{digest}"


def _usage_dict(response: Any) -> dict[str, Any]:
    """Token usage including the cached prompt subset (0 when uncached)."""
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
        "cache_write_tokens": getattr(details, "cache_write_tokens", 0) or 0,
        "total_tokens": usage.total_tokens or 0,
    }


class OpenAILessonAdapter:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4.1", temperature: float = 0.7, max_tokens: int = 8000):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @observe(name="lesson_llm_call", capture_input=True, capture_output=True)
    async def generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Generate lesson - returns (response_text, usage_dict)."""
        logger.info("lesson_api_call", model=self._model, stream=False)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(_SYSTEM_PROMPT),
        )
        text = response.choices[0].message.content or ""
        usage = _usage_dict(response)

        prompt_tokens = usage.get("prompt_tokens", 0)
        logger.info(
            "lesson_api_response",
            tokens_used=usage.get("total_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            cache_hit_ratio=(
                round(usage.get("cached_tokens", 0) / prompt_tokens, 3) if prompt_tokens else 0.0
            ),
        )
        return text, usage

    @observe(name="lesson_llm_call_cached", capture_input=True, capture_output=True)
    async def generate_split(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        """Generate with static system + dynamic user messages for prompt caching.

        The system message carries the static rules (~16K tokens) which OpenAI
        caches automatically; only the small user message varies per request.
        Cache hits cut TTFT by up to 80% and cached input cost by 90%.
        """
        logger.info(
            "lesson_api_call_cached",
            model=self._model,
            system_chars=len(system_prompt),
            user_chars=len(user_prompt),
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(system_prompt),
        )
        text = response.choices[0].message.content or ""
        usage = _usage_dict(response)

        prompt_tokens = usage.get("prompt_tokens", 0)
        logger.info(
            "lesson_api_response_cached",
            tokens_used=usage.get("total_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            cache_hit_ratio=(
                round(usage.get("cached_tokens", 0) / prompt_tokens, 3) if prompt_tokens else 0.0
            ),
        )
        return text, usage

    @observe(name="lesson_llm_stream_cached", capture_input=True, capture_output=True)
    async def stream_generate_split(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncGenerator[tuple[str, dict[str, Any] | None], None]:
        """Stream generation with static system + dynamic user messages.

        Yields (chunk, None) for each text delta, then ("", usage_dict) once
        at the end when the API reports final usage. Same prompt-caching
        benefits as generate_split.
        """
        logger.info(
            "lesson_api_stream_cached",
            model=self._model,
            system_chars=len(system_prompt),
            user_chars=len(user_prompt),
        )
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(system_prompt),
            stream=True,
            stream_options={"include_usage": True},
        )

        final_usage: dict[str, Any] = {}
        async for chunk in stream:
            if chunk.usage:  # final chunk carries usage when include_usage is set
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                final_usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content, None

        prompt_tokens = final_usage.get("prompt_tokens", 0)
        logger.info(
            "lesson_api_stream_cached_done",
            tokens_used=final_usage.get("total_tokens", 0),
            cached_tokens=final_usage.get("cached_tokens", 0),
            cache_hit_ratio=(
                round(final_usage.get("cached_tokens", 0) / prompt_tokens, 3) if prompt_tokens else 0.0
            ),
        )
        yield "", final_usage

    @observe(name="lesson_llm_stream", capture_input=True, capture_output=True)
    async def stream_generate(self, prompt: str) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Stream generate - yields (chunk, usage_dict)."""
        logger.info("lesson_api_call", model=self._model, stream=True)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            prompt_cache_key=_prompt_cache_key(_SYSTEM_PROMPT),
            stream=True,
            stream_options={"include_usage": True},
        )

        # Track usage during streaming
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                # Track completion tokens (approximate)
                completion_tokens += 1
                yield delta.content, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        # At the end, we can't get exact usage from streaming
        # The caller should use the non-streaming version for exact usage
