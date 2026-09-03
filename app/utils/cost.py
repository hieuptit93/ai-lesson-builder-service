# Price per 1M tokens (input, output) in USD
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # GPT-5.6 family (Feb 2026)
    "gpt-5.6-sol":        (4.00,  20.00),
    "gpt-5.6-terra":      (2.00,  12.00),
    "gpt-5.6-luna":       (0.20,   1.20),
    # GPT-4 family
    "gpt-4o":             (2.50,  10.00),
    "gpt-4o-mini":        (0.15,   0.60),
    "gpt-4.1":            (2.00,   8.00),
    "gpt-4.1-mini":       (0.40,   1.60),
}

# Cached-input price per 1M tokens: prompt tokens served FROM the cache.
#
# gpt-5.6-terra is confirmed against the published price sheet ($0.20 against
# $2.00 normal input, i.e. 10%). The other GPT-5.6 entries apply that same 10%
# ratio and are unconfirmed. The GPT-4 family bills cached input at 50%/25%.
_CACHED_INPUT_PRICING: dict[str, float] = {
    "gpt-5.6-sol":        0.40,   # 10% of input - unconfirmed
    "gpt-5.6-terra":      0.20,   # CONFIRMED
    "gpt-5.6-luna":       0.02,   # 10% of input - unconfirmed
    "gpt-4o":             1.25,   # 50% of input
    "gpt-4o-mini":        0.075,  # 50%
    "gpt-4.1":            0.50,   # 25%
    "gpt-4.1-mini":       0.10,   # 25%
}

# Cache-WRITE price per 1M tokens: a cache miss that populates the cache for
# later calls. On the GPT-5.6 family this costs MORE than a plain miss
# ($2.50 vs $2.00 on terra), so the first call is more expensive and the write
# only pays for itself once a later call hits. Models absent here do not bill
# writes separately (the GPT-4 family) - the plain input price applies.
_CACHE_WRITE_PRICING: dict[str, float] = {
    "gpt-5.6-sol":        5.00,   # 125% of input - unconfirmed
    "gpt-5.6-terra":      2.50,   # CONFIRMED
    "gpt-5.6-luna":       0.25,   # 125% of input - unconfirmed
}

_PER_MILLION = 1_000_000


def _input_price(model: str) -> float:
    input_price, _ = _MODEL_PRICING.get(model, _MODEL_PRICING["gpt-4o"])
    return input_price


def cached_input_price(model: str) -> float:
    """Per-1M price for prompt tokens served from the cache."""
    explicit = _CACHED_INPUT_PRICING.get(model)
    return explicit if explicit is not None else _input_price(model)


def cache_write_price(model: str) -> float:
    """Per-1M price for prompt tokens written into the cache.

    Falls back to the normal input price for models that do not bill writes
    separately, which makes a write cost-neutral for them.
    """
    explicit = _CACHE_WRITE_PRICING.get(model)
    return explicit if explicit is not None else _input_price(model)


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Return estimated cost in USD for the given model and token counts.

    `prompt_tokens` is the full input total. It splits three ways:

    - `cached_tokens`       served from cache      -> cached_input_price
    - `cache_write_tokens`  a miss, now cached     -> cache_write_price
    - the remainder         a plain miss           -> normal input price

    Passing only prompt/completion (the default) reproduces full-price
    behaviour. Falls back to gpt-4o pricing for unknown models.
    """
    input_price, output_price = _MODEL_PRICING.get(model, _MODEL_PRICING["gpt-4o"])

    # Clamp so malformed counts can never produce a negative bill.
    cached = max(0, min(cached_tokens, prompt_tokens))
    written = max(0, min(cache_write_tokens, prompt_tokens - cached))
    fresh = prompt_tokens - cached - written

    total = (
        fresh * input_price
        + cached * cached_input_price(model)
        + written * cache_write_price(model)
        + completion_tokens * output_price
    )
    return total / _PER_MILLION


def cache_savings(model: str, cached_tokens: int, cache_write_tokens: int = 0) -> float:
    """Net USD the prompt cache saved on this request.

    Hits save (input - cached) per token; writes cost an extra
    (write - input) per token. The result is NEGATIVE on a first call that
    only populated the cache - that write pays for itself on the next hit.
    """
    input_price = _input_price(model)
    hits = max(0, cached_tokens)
    writes = max(0, cache_write_tokens)

    saved = hits * (input_price - cached_input_price(model))
    overhead = writes * (cache_write_price(model) - input_price)
    return (saved - overhead) / _PER_MILLION


def cache_breakeven_calls(model: str) -> float:
    """How many same-prefix calls make caching cheaper than not caching.

    Solves  write + (n-1) * cached  <=  n * input  for n. A value of 1.3 means
    the second call onward is already cheaper overall.
    """
    input_price = _input_price(model)
    cached = cached_input_price(model)
    write = cache_write_price(model)

    denominator = input_price - cached
    if denominator <= 0:
        return float("inf")  # caching never pays off
    return (write - cached) / denominator
