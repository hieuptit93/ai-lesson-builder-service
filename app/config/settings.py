"""
Application settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # AI Provider
    ai_provider: Literal["openai", "anthropic", "mock"] = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Timeout
    stream_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
