from collections.abc import AsyncGenerator
from typing import Any

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()


class OpenAILessonAdapter:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4.1", temperature: float = 0.7, max_tokens: int = 8000):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Generate lesson - returns (response_text, usage_dict)."""
        logger.info("lesson_api_call", model=self._model, stream=False)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are an expert team of educators creating personalized lessons for children aged 4-8. Always follow the expert discussion format and output valid JSON at the end."},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = response.choices[0].message.content or ""

        # Build usage dict
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        logger.info("lesson_api_response", tokens_used=response.usage.total_tokens if response.usage else 0)
        return text, usage

    async def stream_generate(self, prompt: str) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Stream generate - yields (chunk, usage_dict)."""
        logger.info("lesson_api_call", model=self._model, stream=True)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are an expert team of educators creating personalized lessons for children aged 4-8. Always follow the expert discussion format and output valid JSON at the end."},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
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
