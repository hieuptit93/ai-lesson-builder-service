"""Tests for cost estimation, including prompt-cache hits and writes."""

from app.utils.cost import (
    cache_breakeven_calls,
    cache_savings,
    cache_write_price,
    cached_input_price,
    estimate_cost,
)

# Confirmed gpt-5.6-terra sheet: input $2.00, cached $0.20,
# cache write $2.50, output $12.00 per 1M tokens.
TERRA = "gpt-5.6-terra"


class TestEstimateCostNoCache:
    def test_known_model(self):
        assert estimate_cost("gpt-4.1", prompt_tokens=1_000_000, completion_tokens=0) == 2.00
        assert estimate_cost("gpt-4.1", prompt_tokens=0, completion_tokens=1_000_000) == 8.00

    def test_terra_matches_price_sheet(self):
        assert estimate_cost(TERRA, 1_000_000, 0) == 2.00
        assert estimate_cost(TERRA, 0, 1_000_000) == 12.00

    def test_unknown_model_falls_back_to_gpt4o(self):
        assert estimate_cost("some-future-model", 1_000_000, 0) == estimate_cost("gpt-4o", 1_000_000, 0)

    def test_zero_tokens_is_free(self):
        assert estimate_cost(TERRA, 0, 0) == 0.0

    def test_defaults_preserve_full_price_behaviour(self):
        assert estimate_cost(TERRA, 10_000, 2_000) == estimate_cost(
            TERRA, 10_000, 2_000, cached_tokens=0, cache_write_tokens=0
        )


class TestCacheHits:
    def test_fully_cached_uses_cached_rate(self):
        # 1M tokens all served from cache -> $0.20
        assert estimate_cost(TERRA, 1_000_000, 0, cached_tokens=1_000_000) == 0.20

    def test_cached_is_ten_percent_of_input_on_terra(self):
        assert cached_input_price(TERRA) == 0.20

    def test_partial_cache_splits_the_bill(self):
        # 50% cached: 500k * $2.00 + 500k * $0.20 = 1.00 + 0.10
        cost = estimate_cost(TERRA, 1_000_000, 0, cached_tokens=500_000)
        assert abs(cost - 1.10) < 1e-9

    def test_hit_is_cheaper_than_miss(self):
        miss = estimate_cost(TERRA, 100_000, 0)
        hit = estimate_cost(TERRA, 100_000, 0, cached_tokens=100_000)
        assert hit < miss

    def test_completion_tokens_never_discounted(self):
        a = estimate_cost(TERRA, 1_000, 1_000_000, cached_tokens=1_000)
        b = estimate_cost(TERRA, 1_000, 1_000_000, cached_tokens=0)
        assert abs((b - a) - 1_000 * (2.00 - 0.20) / 1_000_000) < 1e-12


class TestCacheWrites:
    def test_write_costs_more_than_plain_miss_on_terra(self):
        # This is the counter-intuitive part: populating the cache is dearer.
        plain = estimate_cost(TERRA, 1_000_000, 0)
        write = estimate_cost(TERRA, 1_000_000, 0, cache_write_tokens=1_000_000)
        assert write > plain
        assert write == 2.50

    def test_write_price_lookup(self):
        assert cache_write_price(TERRA) == 2.50

    def test_gpt4_family_write_is_cost_neutral(self):
        # Not billed separately -> same as a plain miss
        plain = estimate_cost("gpt-4o", 1_000_000, 0)
        write = estimate_cost("gpt-4o", 1_000_000, 0, cache_write_tokens=1_000_000)
        assert write == plain

    def test_three_way_split(self):
        # 100k cached + 100k written + 800k fresh
        cost = estimate_cost(
            TERRA, 1_000_000, 0, cached_tokens=100_000, cache_write_tokens=100_000
        )
        expected = (100_000 * 0.20 + 100_000 * 2.50 + 800_000 * 2.00) / 1_000_000
        assert abs(cost - expected) < 1e-9


class TestClamping:
    def test_cached_exceeding_prompt_is_clamped(self):
        assert estimate_cost(TERRA, 1_000, 0, cached_tokens=999_999) == estimate_cost(
            TERRA, 1_000, 0, cached_tokens=1_000
        )

    def test_write_clamped_to_uncached_remainder(self):
        # cached already covers everything, so the write must contribute nothing
        both = estimate_cost(TERRA, 1_000, 0, cached_tokens=1_000, cache_write_tokens=1_000)
        assert both == estimate_cost(TERRA, 1_000, 0, cached_tokens=1_000)

    def test_negative_counts_treated_as_zero(self):
        assert estimate_cost(TERRA, 1_000, 0, cached_tokens=-50, cache_write_tokens=-9) == \
            estimate_cost(TERRA, 1_000, 0)

    def test_never_negative(self):
        assert estimate_cost(TERRA, 0, 0, cached_tokens=500, cache_write_tokens=500) >= 0.0


class TestCacheSavings:
    def test_no_activity_no_savings(self):
        assert cache_savings(TERRA, 0) == 0.0

    def test_hits_save_the_price_delta(self):
        # $2.00 - $0.20 = $1.80 per 1M
        assert abs(cache_savings(TERRA, 1_000_000) - 1.80) < 1e-9

    def test_write_only_call_is_a_net_loss(self):
        # First call populates the cache and costs $0.50/1M extra
        assert cache_savings(TERRA, 0, cache_write_tokens=1_000_000) < 0
        assert abs(cache_savings(TERRA, 0, cache_write_tokens=1_000_000) + 0.50) < 1e-9

    def test_gpt4_write_has_no_overhead(self):
        assert cache_savings("gpt-4o", 0, cache_write_tokens=1_000_000) == 0.0

    def test_savings_reconcile_with_estimate(self):
        prompt, cached, written = 80_000, 50_000, 10_000
        full = estimate_cost(TERRA, prompt, 0)
        actual = estimate_cost(TERRA, prompt, 0, cached_tokens=cached, cache_write_tokens=written)
        assert abs((full - actual) - cache_savings(TERRA, cached, written)) < 1e-12


class TestBreakeven:
    def test_terra_pays_off_on_second_call(self):
        # (2.50 - 0.20) / (2.00 - 0.20) = 1.277...
        n = cache_breakeven_calls(TERRA)
        assert 1.0 < n < 2.0

    def test_two_calls_beat_no_cache_on_terra(self):
        tokens = 1_000_000
        no_cache = 2 * estimate_cost(TERRA, tokens, 0)
        with_cache = (
            estimate_cost(TERRA, tokens, 0, cache_write_tokens=tokens)  # write
            + estimate_cost(TERRA, tokens, 0, cached_tokens=tokens)     # hit
        )
        assert with_cache < no_cache

    def test_gpt4_family_pays_off_immediately(self):
        assert cache_breakeven_calls("gpt-4o") <= 1.0


class TestPriceTableIntegrity:
    MODELS = (
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
    )

    def test_cached_never_exceeds_input(self):
        for model in self.MODELS:
            full = estimate_cost(model, 1_000_000, 0)
            cached = estimate_cost(model, 1_000_000, 0, cached_tokens=1_000_000)
            assert cached <= full, model

    def test_every_model_has_finite_breakeven(self):
        for model in self.MODELS:
            assert cache_breakeven_calls(model) < float("inf"), model
