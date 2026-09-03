"""Tests for the real OpenAIVisionAdapter.

These construct the ACTUAL adapter rather than a fake. A regression guard:
the image-download caches live in __init__, and an edit that displaced them
(properties inserted mid-__init__) shipped an adapter whose every request
failed with "object has no attribute '_variant_cache'" - no fake-based test
could have caught it.
"""

import asyncio

import pytest
from openai import AsyncOpenAI

from app.domains.vision_extract.infrastructure.openai_vision_adapter import (
    OpenAIVisionAdapter,
    _prompt_cache_key,
    _resize_image_sync,
)

# Every attribute the adapter's methods rely on being set by __init__.
REQUIRED_ATTRS = (
    "_client",
    "_model",
    "_guardrail_model",
    "_suggestions_model",
    "_temperature",
    "_max_tokens",
    "_http",
    "_download_tasks",
    "_variant_cache",
)


def _adapter(**kwargs) -> OpenAIVisionAdapter:
    defaults = dict(
        client=AsyncOpenAI(api_key="test-key-not-used"),
        model="gpt-5.6-terra",
        guardrail_model="gpt-5.6-luna",
        suggestions_model="gpt-5.6-luna",
    )
    defaults.update(kwargs)
    return OpenAIVisionAdapter(**defaults)


class TestInitialization:
    @pytest.mark.parametrize("attr", REQUIRED_ATTRS)
    def test_attribute_is_set(self, attr):
        assert hasattr(_adapter(), attr), f"__init__ did not set {attr}"

    def test_caches_start_empty(self):
        adapter = _adapter()
        assert adapter._download_tasks == {}
        assert adapter._variant_cache == {}
        assert adapter._http is None

    def test_minimal_construction_works(self):
        # Only the client is truly required; the rest must have defaults.
        adapter = OpenAIVisionAdapter(client=AsyncOpenAI(api_key="k"))
        for attr in REQUIRED_ATTRS:
            assert hasattr(adapter, attr), attr

    def test_suggestions_model_defaults_to_model(self):
        adapter = OpenAIVisionAdapter(client=AsyncOpenAI(api_key="k"), model="gpt-5.6-terra")
        assert adapter.suggestions_model == "gpt-5.6-terra"

    def test_suggestions_model_override(self):
        assert _adapter(suggestions_model="gpt-5.6-luna").suggestions_model == "gpt-5.6-luna"


class TestModelProperties:
    def test_properties_match_constructor(self):
        adapter = _adapter()
        assert adapter.model == "gpt-5.6-terra"
        assert adapter.suggestions_model == "gpt-5.6-luna"
        assert adapter.guardrail_model == "gpt-5.6-luna"

    def test_properties_are_read_only(self):
        with pytest.raises(AttributeError):
            _adapter().model = "something-else"


class TestImageCache:
    """Exercises the paths that touch _variant_cache / _download_tasks."""

    PNG_1X1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    def test_clear_image_cache_on_fresh_adapter(self):
        _adapter().clear_image_cache()  # must not raise

    def test_get_image_populates_cache(self, monkeypatch):
        adapter = _adapter()

        async def fake_download(url):
            return self.PNG_1X1

        monkeypatch.setattr(adapter, "_download_raw", fake_download)
        result = asyncio.run(adapter._get_image("http://x/a.png", "full"))

        assert result.startswith("data:")
        assert ("http://x/a.png", "full") in adapter._variant_cache

    def test_second_call_hits_cache_without_redownloading(self, monkeypatch):
        adapter = _adapter()
        calls = []

        async def fake_download(url):
            calls.append(url)
            return self.PNG_1X1

        monkeypatch.setattr(adapter, "_download_raw", fake_download)

        async def run_twice():
            a = await adapter._get_image("http://x/a.png", "full")
            b = await adapter._get_image("http://x/a.png", "full")
            return a, b

        first, second = asyncio.run(run_twice())
        assert first == second
        assert len(calls) == 1, "cached variant should not re-download"

    def test_variants_cached_separately(self, monkeypatch):
        adapter = _adapter()

        async def fake_download(url):
            return self.PNG_1X1

        monkeypatch.setattr(adapter, "_download_raw", fake_download)

        async def run_both():
            await adapter._get_image("http://x/a.png", "full")
            await adapter._get_image("http://x/a.png", "low")

        asyncio.run(run_both())
        assert ("http://x/a.png", "full") in adapter._variant_cache
        assert ("http://x/a.png", "low") in adapter._variant_cache

    def test_failed_download_falls_back_to_url(self, monkeypatch):
        adapter = _adapter()

        async def failing_download(url):
            return None

        monkeypatch.setattr(adapter, "_download_raw", failing_download)
        result = asyncio.run(adapter._get_image("http://x/gone.png", "full"))
        assert result == "http://x/gone.png"

    def test_clear_after_use_empties_caches(self, monkeypatch):
        adapter = _adapter()

        async def fake_download(url):
            return self.PNG_1X1

        monkeypatch.setattr(adapter, "_download_raw", fake_download)
        asyncio.run(adapter._get_image("http://x/a.png", "full"))
        adapter.clear_image_cache()
        assert adapter._variant_cache == {}
        assert adapter._download_tasks == {}

    def test_build_image_content_shape(self, monkeypatch):
        adapter = _adapter()

        async def fake_download(url):
            return self.PNG_1X1

        monkeypatch.setattr(adapter, "_download_raw", fake_download)
        items = asyncio.run(adapter._build_image_content(["http://x/a.png"], "low"))
        assert items[0]["type"] == "input_image"
        assert items[0]["detail"] == "low"


class TestHelpers:
    def test_prompt_cache_key_is_stable(self):
        assert _prompt_cache_key("abc") == _prompt_cache_key("abc")

    def test_prompt_cache_key_differs_per_prompt(self):
        assert _prompt_cache_key("abc") != _prompt_cache_key("xyz")

    def test_resize_falls_back_on_undecodable_bytes(self):
        raw = b"definitely not an image"
        out, mime = _resize_image_sync(raw, 512)
        assert out == raw
        assert mime == "image/jpeg"


class TestUsageDict:
    class _Details:
        cached_tokens = 600
        cache_write_tokens = 0

    class _Usage:
        input_tokens = 1000
        output_tokens = 200
        input_tokens_details = None

    def test_reads_cached_tokens(self):
        usage = self._Usage()
        usage.input_tokens_details = self._Details()

        class R:
            pass

        response = R()
        response.usage = usage
        result = OpenAIVisionAdapter._usage_dict(response)
        assert result["cached_tokens"] == 600
        assert result["prompt_tokens"] == 1000
        assert result["total_tokens"] == 1200

    def test_missing_details_defaults_to_zero(self):
        class R:
            pass

        response = R()
        response.usage = self._Usage()  # input_tokens_details is None
        result = OpenAIVisionAdapter._usage_dict(response)
        assert result["cached_tokens"] == 0
        assert result["cache_write_tokens"] == 0

    def test_no_usage_returns_empty(self):
        class R:
            usage = None

        assert OpenAIVisionAdapter._usage_dict(R()) == {}
