"""Tests for chat API behavior."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langsmith-key")

from api.dependencies import AppContainer  # noqa: E402
from api.main import create_app  # noqa: E402
from core.settings import Settings  # noqa: E402
from services.spacex_client_interface import SpaceXClientInterface  # noqa: E402


class FakeRunner:
    """Fake agent runner that returns deterministic output."""

    async def run(self, *, user_message: str, thread_id: str) -> str:
        return f"thread={thread_id} reply={user_message}"


class FakeSpaceXClient(SpaceXClientInterface):
    """No-op client for API tests."""

    async def get_latest_launch(self) -> dict:
        return {}

    async def get_next_launch(self) -> dict:
        return {}

    async def query_launches(
        self,
        query: dict,
        *,
        options: dict | None = None,
    ) -> dict:
        return {"docs": [], "totalDocs": 0}

    async def get_rockets(self) -> list[dict]:
        return []

    async def get_launchpads(self) -> list[dict]:
        return []

    async def close(self) -> None:
        return None


def _build_test_container() -> AppContainer:
    settings = Settings(
        openai_api_key="test-openai-key",
        langchain_api_key="test-langsmith-key",
        langchain_tracing_v2=False,
        openai_model="gpt-4.1-mini",
        spacex_api_base_url="https://api.spacexdata.com/v4",
        max_user_message_chars=30,
    )
    return AppContainer(
        settings=settings,
        spacex_client=FakeSpaceXClient(),
        agent_runner=FakeRunner(),
    )


def test_chat_endpoint_generates_thread_id() -> None:
    app = create_app(container_builder=_build_test_container)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["thread_id"]
        assert "reply=hello" in payload["reply"]


def test_chat_endpoint_reuses_thread_id() -> None:
    app = create_app(container_builder=_build_test_container)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hello", "thread_id": "abc-123"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["thread_id"] == "abc-123"


def test_chat_endpoint_enforces_max_length() -> None:
    app = create_app(container_builder=_build_test_container)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "x" * 100})
        assert response.status_code == 422
