class AIServiceError(Exception):
    def __init__(self, message: str, error_code: str = "internal_error"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class UnsafeContentError(AIServiceError):
    def __init__(self, message: str, flagged_items: list | None = None):
        super().__init__(message, error_code="unsafe_content")
        self.flagged_items = flagged_items or []


class ExtractionError(AIServiceError):
    def __init__(self, message: str):
        super().__init__(message, error_code="extraction_failed")


class LowQualityImageError(AIServiceError):
    def __init__(self, message: str, confidence: float = 0.0):
        super().__init__(message, error_code="image_quality_low")
        self.confidence = confidence


class LLMProviderError(AIServiceError):
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, error_code="llm_provider_error")
        self.provider = provider


class MemoryServiceError(AIServiceError):
    def __init__(self, message: str):
        super().__init__(message, error_code="memory_unavailable")


class RateLimitExceededError(AIServiceError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, error_code="rate_limit_exceeded")


class MissingImageUrlsError(AIServiceError):
    def __init__(self):
        super().__init__(
            "At least one image URL is required for v3 lesson generation",
            error_code="missing_image_urls",
        )


class VisionExtractionError(AIServiceError):
    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message, error_code="vision_extraction_failed")
        self.raw_response = raw_response
