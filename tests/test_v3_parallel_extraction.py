"""Tests for v3 parallel per-image extraction and result merging."""

import asyncio

import pytest

from app.domains.vision_extract.application.services.extraction_service import (
    ExtractionService,
    _batch,
    _renumber_lesson_titles,
)


class FakeAdapter:
    """Stands in for OpenAIVisionAdapter, recording calls per batch."""

    model = "gpt-5.6-terra"
    suggestions_model = "gpt-5.6-luna"
    guardrail_model = "gpt-5.6-luna"

    def __init__(self, responses=None, error_on=None, delay=0.0):
        # responses: dict keyed by the tuple of urls in the batch
        self._responses = responses or {}
        self._error_on = error_on or set()
        self._delay = delay
        self.calls: list[tuple[str, ...]] = []
        self.max_concurrent = 0
        self._active = 0

    async def extract_v3_suggestions(self, image_urls, prompt):
        key = tuple(image_urls)
        self.calls.append(key)
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if key in self._error_on:
                raise RuntimeError(f"batch failed: {key}")
            return self._responses.get(key, self._default(key))
        finally:
            self._active -= 1

    @staticmethod
    def _default(key):
        return (
            {
                "rejected": False,
                "reason_code": None,
                "reason": "",
                "suggested_lessons": [
                    {"title": "Lesson 1: From " + key[0], "agent_mode": "learn_agent",
                     "content": "c", "options": []}
                ],
            },
            {"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 0,
             "cache_write_tokens": 0, "total_tokens": 150},
        )


def _service(adapter, **kwargs):
    defaults = dict(v3_parallel_images=True, v3_images_per_batch=1, v3_max_concurrent=4)
    defaults.update(kwargs)
    return ExtractionService(adapter=adapter, **defaults)


class TestBatch:
    def test_splits_evenly(self):
        assert _batch(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]

    def test_last_batch_partial(self):
        assert _batch(["a", "b", "c"], 2) == [["a", "b"], ["c"]]

    def test_size_one_is_full_fanout(self):
        assert _batch(["a", "b"], 1) == [["a"], ["b"]]

    def test_size_larger_than_input(self):
        assert _batch(["a"], 5) == [["a"]]

    def test_zero_size_treated_as_one(self):
        assert _batch(["a", "b"], 0) == [["a"], ["b"]]


class TestRenumberTitles:
    def test_renumbers_merged_duplicates(self):
        lessons = [
            {"title": "Lesson 1: Pronouns"},
            {"title": "Lesson 2: Verbs"},
            {"title": "Lesson 1: Reading"},   # from the second batch
        ]
        _renumber_lesson_titles(lessons)
        assert [l["title"] for l in lessons] == [
            "Lesson 1: Pronouns",
            "Lesson 2: Verbs",
            "Lesson 3: Reading",
        ]

    def test_vietnamese_keyword(self):
        lessons = [{"title": "Bài 1: Từ vựng"}, {"title": "Bài 1: Ngữ pháp"}]
        _renumber_lesson_titles(lessons)
        assert [l["title"] for l in lessons] == ["Bài 1: Từ vựng", "Bài 2: Ngữ pháp"]

    def test_title_without_prefix_untouched(self):
        lessons = [{"title": "Fill in the Blanks"}, {"title": "Lesson 1: Verbs"}]
        _renumber_lesson_titles(lessons)
        assert lessons[0]["title"] == "Fill in the Blanks"
        assert lessons[1]["title"] == "Lesson 2: Verbs"

    def test_preserves_topic_text(self):
        lessons = [{"title": "Lesson 7: Match Sentences and Pictures"}]
        _renumber_lesson_titles(lessons)
        assert lessons[0]["title"] == "Lesson 1: Match Sentences and Pictures"

    def test_missing_title_does_not_raise(self):
        lessons = [{}, {"title": None}]
        _renumber_lesson_titles(lessons)  # must not raise


@pytest.mark.asyncio
class TestParallelExtraction:
    async def test_single_image_uses_one_call(self):
        adapter = FakeAdapter()
        result, usage = await _service(adapter).extract_suggestions_v3(["img1"])
        assert adapter.calls == [("img1",)]
        assert len(result["suggested_lessons"]) == 1

    async def test_multi_image_fans_out(self):
        adapter = FakeAdapter()
        result, usage = await _service(adapter).extract_suggestions_v3(["img1", "img2", "img3"])
        assert adapter.calls == [("img1",), ("img2",), ("img3",)]
        assert len(result["suggested_lessons"]) == 3

    async def test_parallel_disabled_makes_one_call(self):
        adapter = FakeAdapter()
        svc = _service(adapter, v3_parallel_images=False)
        await svc.extract_suggestions_v3(["img1", "img2"])
        assert adapter.calls == [("img1", "img2")]

    async def test_batch_size_groups_images(self):
        adapter = FakeAdapter()
        svc = _service(adapter, v3_images_per_batch=2)
        await svc.extract_suggestions_v3(["a", "b", "c"])
        assert adapter.calls == [("a", "b"), ("c",)]

    async def test_calls_actually_run_concurrently(self):
        adapter = FakeAdapter(delay=0.05)
        svc = _service(adapter)
        await svc.extract_suggestions_v3(["a", "b", "c", "d"])
        assert adapter.max_concurrent > 1

    async def test_concurrency_is_capped(self):
        adapter = FakeAdapter(delay=0.02)
        svc = _service(adapter, v3_max_concurrent=2)
        await svc.extract_suggestions_v3(["a", "b", "c", "d", "e"])
        assert adapter.max_concurrent <= 2

    async def test_usage_is_summed(self):
        adapter = FakeAdapter()
        _, usage = await _service(adapter).extract_suggestions_v3(["a", "b"])
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 100
        assert usage["total_tokens"] == 300

    async def test_lesson_order_follows_image_order(self):
        adapter = FakeAdapter(delay=0.01)
        result, _ = await _service(adapter).extract_suggestions_v3(["p1", "p2", "p3"])
        titles = [l["title"] for l in result["suggested_lessons"]]
        assert "p1" in titles[0] and "p2" in titles[1] and "p3" in titles[2]

    async def test_titles_renumbered_across_batches(self):
        adapter = FakeAdapter()
        result, _ = await _service(adapter).extract_suggestions_v3(["a", "b", "c"])
        numbers = [l["title"].split(":")[0] for l in result["suggested_lessons"]]
        assert numbers == ["Lesson 1", "Lesson 2", "Lesson 3"]


@pytest.mark.asyncio
class TestParallelFailureHandling:
    async def test_one_failed_batch_keeps_the_rest(self):
        adapter = FakeAdapter(error_on={("b",)})
        result, _ = await _service(adapter).extract_suggestions_v3(["a", "b", "c"])
        assert len(result["suggested_lessons"]) == 2
        assert result["rejected"] is False

    async def test_all_batches_failing_raises(self):
        adapter = FakeAdapter(error_on={("a",), ("b",)})
        with pytest.raises(RuntimeError):
            await _service(adapter).extract_suggestions_v3(["a", "b"])

    async def test_all_rejected_returns_rejection(self):
        rejected = (
            {"rejected": True, "reason_code": "no_educational_content",
             "reason": "blank page", "suggested_lessons": []},
            {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        )
        adapter = FakeAdapter(responses={("a",): rejected, ("b",): rejected})
        result, usage = await _service(adapter).extract_suggestions_v3(["a", "b"])
        assert result["rejected"] is True
        assert result["reason_code"] == "no_educational_content"
        assert usage["prompt_tokens"] == 20  # usage still accounted

    async def test_partial_rejection_keeps_good_lessons(self):
        rejected = (
            {"rejected": True, "reason_code": "no_educational_content",
             "reason": "blank", "suggested_lessons": []},
            {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        )
        adapter = FakeAdapter(responses={("b",): rejected})
        result, _ = await _service(adapter).extract_suggestions_v3(["a", "b"])
        assert result["rejected"] is False
        assert len(result["suggested_lessons"]) == 1

    async def test_no_images_short_circuits(self):
        adapter = FakeAdapter()
        result, usage = await _service(adapter).extract_suggestions_v3([])
        assert result["suggested_lessons"] == []
        assert adapter.calls == []


class TestModelLabel:
    def test_defaults_to_adapter_model_not_hardcoded_gpt4o(self):
        svc = ExtractionService(adapter=FakeAdapter())
        assert svc._model_name == "gpt-5.6-terra"

    def test_suggestions_label_uses_suggestions_model(self):
        svc = ExtractionService(adapter=FakeAdapter())
        assert svc._suggestions_model_name == "gpt-5.6-luna"

    def test_explicit_label_still_honoured(self):
        svc = ExtractionService(adapter=FakeAdapter(), model_name="custom-label")
        assert svc._model_name == "custom-label"
