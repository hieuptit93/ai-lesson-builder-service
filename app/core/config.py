import os

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Override __init__ to export Langfuse config to os.environ on first load
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Export Langfuse config to os.environ so Langfuse SDK can read it
        # Langfuse SDK v3 reads from os.environ["LANGFUSE_*"]
        if self.langfuse_secret_key:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", self.langfuse_secret_key)
        if self.langfuse_public_key:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key)
        if self.langfuse_base_url:
            os.environ.setdefault("LANGFUSE_BASE_URL", self.langfuse_base_url)

    app_name: str = "ai-lesson-builder-service"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 30001
    log_level: str = "INFO"

    # Extra settings from .env (backwards compatibility)
    ai_provider: str = "openai"
    stream_timeout: int = 60

    openai_api_key: str = ""
    # GPT-5.6 Terra: $2/1M input, 128K output, Feb 2026 knowledge
    # Using Structured Outputs with strict schema - decoder enforced compliance
    openai_vision_model: str = "gpt-5.6-terra"
    # GPT-5.6 Luna: $0.20/1M input - ultra cheap for guardrail
    openai_guardrail_model: str = "gpt-5.6-luna"
    # Enable guardrail for custom "educational value" check
    openai_guardrail_enabled: bool = True
    openai_lesson_model: str = "gpt-4.1"
    # v3/lessons/generate. Kept as its own knob so this endpoint can be tuned
    # independently, but it must stay a strong vision model: generate is the
    # ONLY step that sees the images, and generate_artifact works purely from
    # the `content` text it produces (GenerateArtifactRequest has no
    # image_urls). A transcription error here is unrecoverable downstream -
    # e.g. misreading "12 / 5" makes artifact pre-solve the WRONG problem
    # correctly, with no signal that anything went wrong.
    # A smaller model was measured as faster but is not worth that ceiling.
    openai_suggestions_model: str = "gpt-5.6-terra"
    openai_vision_temperature: float = 0.2
    openai_lesson_temperature: float = 0.7

    # v3 multi-image requests are split into parallel calls: each call emits
    # fewer output tokens, so wall-clock drops to roughly max() instead of
    # sum(). Disable if lessons spanning two pages get split incorrectly.
    openai_v3_parallel_images: bool = True
    openai_v3_images_per_batch: int = 1   # images per call; raise to reduce page-boundary splits
    openai_v3_max_concurrent: int = 4     # cap fan-out so we don't trip rate limits
    # V3 single-call generates ALL lessons' full fields (content + summary +
    # detail_tasks_lesson + prompt_agent) in one response. 5 dense pages can
    # need 8-15 lessons x 400-800 tokens each. GPT-5.6 Terra supports 128K
    # output; this is a cap, cost is only charged for tokens actually generated.
    openai_vision_max_tokens: int = 16000
    openai_lesson_max_tokens: int = 8000

    mem0_base_url: str = ""
    mem0_timeout: float = 5.0

    # Profile API (external)
    profile_api_base_url: str = ""
    profile_api_token: str = ""

    # Langfuse (LLM Observability) - for tracing via @observe decorator
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_sample_rate: float = 1.0
    langfuse_tracing_enabled: bool = True

    # Delay between SSE phases (seconds). Default 0.0 for production.
    # Set > 0 only for debugging/demo to visualize progress.
    stream_delay_seconds: float = 0.0

    # Expert Avatar URLs for SSE streaming (5 AI agents)
    expert_avatar_vision_analyst: str = "https://smedia.stepup.edu.vn/robot/game/asset/A_vision_analyst_512.png"
    expert_avatar_curriculum_designer: str = "https://smedia.stepup.edu.vn/robot/game/asset/B_curriculum_designer_512.png"
    expert_avatar_child_psychologist: str = "https://smedia.stepup.edu.vn/robot/game/asset/C_child_psychologist_512.png"
    expert_avatar_safety_reviewer: str = "https://smedia.stepup.edu.vn/robot/game/asset/D_safety_reviewer_512.png"
    expert_avatar_final_editor: str = "https://smedia.stepup.edu.vn/robot/game/asset/E_final_editor_512.png"

    # Pipeline robot image for SSE events
    pipeline_robot_image_url: str = "https://smedia.stepup.edu.vn/thecoach/all_files/pika_robot.png"

    # API Key Authentication
    # Enable/disable API key validation (dev vs prod)
    x_api_key_enabled: bool = False
    # The API key that clients must provide in X-API-Key header
    x_api_key: str = ""

    # Swagger UI Authentication (HTTP Basic Auth)
    # Override via SWAGGER_USERNAME / SWAGGER_PASSWORD in .env
    swagger_username: str = "admin"
    swagger_password: str = "pika-admin"

    # Google Chat Alerting
    # Webhook URL for sending alerts to Google Chat
    google_chat_webhook_url: str = ""
    # Enable/disable alerting (set to False to suppress all alerts)
    alerting_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
