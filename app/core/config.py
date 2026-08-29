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
    openai_vision_temperature: float = 0.2
    openai_lesson_temperature: float = 0.7
    openai_vision_max_tokens: int = 4000
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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
