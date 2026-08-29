# Langfuse SDK initialization - singleton pattern for SDK v3
#
# Usage:
#   from app.core.tracing import get_langfuse_client
#   langfuse = get_langfuse_client()
#
# The @observe decorator automatically traces async functions.
# Just add @observe(name="my_function") to any async function.
#
# Example:
#   from langfuse import observe
#
#   @observe(name="generate_lesson")
#   async def generate_lesson():
#       # Automatically creates a trace in Langfuse
#       ...
#       langfuse = get_langfuse_client()
#       langfuse.get_current_observation().update(metadata={"key": "value"})

from typing import TYPE_CHECKING, Optional

import structlog

from app.core.config import Settings

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = structlog.get_logger()

# Langfuse client singleton - initialized on first use
_langfuse_client: Optional["Langfuse"] = None


def get_langfuse_client() -> Optional["Langfuse"]:
    """Get or create the Langfuse client singleton.

    Returns None if Langfuse keys are not configured.
    This uses get_client() from SDK v3 which is the recommended approach.
    """
    global _langfuse_client

    if _langfuse_client is not None:
        return _langfuse_client

    settings = Settings()

    # Check if Langfuse is configured
    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.debug(
            "langfuse.skipped",
            reason="missing_api_keys",
            has_secret=bool(settings.langfuse_secret_key),
            has_public=bool(settings.langfuse_public_key),
        )
        return None

    try:
        from langfuse import Langfuse

        # Langfuse() initializes the singleton (keyed by public_key) with
        # tracing_enabled=False — disables OTel BatchSpanProcessor and its
        # background thread, eliminating GIL contention spikes.
        # Prompt fetching (get_prompt / prompt_cache) is unaffected.
        # Subsequent get_client() calls in other modules reuse this singleton
        # via _create_client_from_instance(), which forwards tracing_enabled.
        _langfuse_client = Langfuse(tracing_enabled=settings.langfuse_tracing_enabled)

        logger.info(
            "langfuse.initialized",
            base_url=settings.langfuse_base_url,
            sample_rate=settings.langfuse_sample_rate,
        )

        return _langfuse_client

    except Exception as e:
        logger.warning(
            "langfuse.init_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


def flush_langfuse():
    """Flush pending Langfuse events.

    Call this in application shutdown or in short-lived processes.
    """
    try:
        client = get_langfuse_client()
        if client:
            client.flush()
    except Exception as e:
        logger.warning("langfuse.flush_failed", error=str(e))
