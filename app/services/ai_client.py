"""
AI Client - Unified interface for OpenAI and Anthropic.
"""
import base64
import httpx
from typing import AsyncGenerator, Optional
from loguru import logger

from app.config import settings


class AIClient:
    """Unified AI client supporting OpenAI, Anthropic, and Mock."""

    def __init__(self):
        self.provider = settings.ai_provider

        if self.provider == "mock":
            # Use mock client for testing
            from app.services.ai_client_mock import MockAIClient
            self._mock = MockAIClient()
            self.client = None
            self.model = "mock"
            logger.info("🧪 Using MOCK AI Client (no real API calls)")
        elif self.provider == "openai":
            from openai import AsyncOpenAI
            self._mock = None
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_model
        else:
            from anthropic import AsyncAnthropic
            self._mock = None
            self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            self.model = settings.anthropic_model

    async def analyze_image(
        self,
        image_urls: list[str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze images with vision model.
        Returns the text response.
        """
        logger.info(f"Analyzing {len(image_urls)} images with {self.provider}")

        if self._mock:
            return await self._mock.analyze_image(image_urls, prompt, system_prompt)
        elif self.provider == "openai":
            return await self._openai_vision(image_urls, prompt, system_prompt)
        else:
            return await self._anthropic_vision(image_urls, prompt, system_prompt)

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text completion."""
        logger.info(f"Generating text with {self.provider}")

        if self._mock:
            return await self._mock.generate_text(prompt, system_prompt)
        elif self.provider == "openai":
            return await self._openai_text(prompt, system_prompt)
        else:
            return await self._anthropic_text(prompt, system_prompt)

    async def generate_text_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text completion with streaming."""
        logger.info(f"Streaming text with {self.provider}")

        if self._mock:
            async for chunk in self._mock.generate_text_stream(prompt, system_prompt):
                yield chunk
        elif self.provider == "openai":
            async for chunk in self._openai_stream(prompt, system_prompt):
                yield chunk
        else:
            async for chunk in self._anthropic_stream(prompt, system_prompt):
                yield chunk

    # ═══════════════════════════════════════════════════════════════════════
    # OpenAI Implementation
    # ═══════════════════════════════════════════════════════════════════════

    async def _openai_vision(
        self,
        image_urls: list[str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """OpenAI GPT-4V vision analysis."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build content with images
        content = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "high"}
            })

        messages.append({"role": "user", "content": content})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )

        return response.choices[0].message.content

    async def _openai_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """OpenAI text completion."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )

        return response.choices[0].message.content

    async def _openai_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """OpenAI streaming completion."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ═══════════════════════════════════════════════════════════════════════
    # Anthropic Implementation
    # ═══════════════════════════════════════════════════════════════════════

    async def _anthropic_vision(
        self,
        image_urls: list[str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Anthropic Claude vision analysis."""
        content = []

        # Add images
        for url in image_urls:
            # Anthropic requires base64 for images, or we can use URL
            content.append({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": url,
                }
            })

        content.append({"type": "text", "text": prompt})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text

    async def _anthropic_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Anthropic text completion."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    async def _anthropic_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Anthropic streaming completion."""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


# Singleton instance
ai_client = AIClient()
