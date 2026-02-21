"""Pytest shared fixtures."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from core.settings import get_settings


@pytest.fixture(autouse=True)
def set_required_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Provide required environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("SPACEX_API_BASE_URL", "https://api.spacexdata.com/v5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
