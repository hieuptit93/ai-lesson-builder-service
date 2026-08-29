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

_PER_MILLION = 1_000_000


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated cost in USD for the given model and token counts.

    Uses a static price table (input price, output price per 1M tokens).
    Falls back to gpt-4o pricing for unknown models.
    """
    input_price, output_price = _MODEL_PRICING.get(model, _MODEL_PRICING["gpt-4o"])
    return (prompt_tokens * input_price + completion_tokens * output_price) / _PER_MILLION
