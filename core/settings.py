"""Application settings management."""

from __future__ import annotations

from functools import lru_cache

from pydantic import (
    AnyHttpUrl,
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Typed runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "SpaceX Agent API"
    openai_api_key: SecretStr = Field(..., description="OpenAI API key.")
    langchain_api_key: SecretStr = Field(..., description="LangSmith API key.")
    langchain_tracing_v2: bool = True
    langchain_project: str = "space-x-agent"
    openai_model: str = "gpt-5-mini"
    spacex_api_base_url: AnyHttpUrl = AnyHttpUrl("https://api.spacexdata.com/v4")
    request_timeout_seconds: PositiveFloat = 10.0
    max_history_messages: PositiveInt = 20
    max_user_message_chars: PositiveInt = 2000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        """Normalize CORS origins from either a list or a comma-separated string.

        Args:
            value: Raw input value from environment/settings parsing.

        Returns:
            A normalized list of origin strings.
        """
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ValueError("CORS_ORIGINS must be a list or comma-separated string.")


@lru_cache
def get_settings() -> Settings:
    """Load and cache validated settings.

    Returns:
        Application settings.

    Raises:
        ConfigurationError: If required environment variables are missing or invalid.
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid application configuration: {exc}") from exc
