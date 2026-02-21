"""Tests for chat API behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Literal

from fastapi.testclient import TestClient

from api.dependencies import AppContainer

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langsmith-key")

from api.main import create_app
from core.settings import Settings
from services.spacex_client_interface import (
    LaunchRecord,
    QueryResponse,
    SpaceXClientInterface,
)


class FakeRunner:
    """Fake agent runner that returns deterministic output."""

    async def run(self, *, user_message: str, thread_id: str) -> str:
        return f"thread={thread_id} reply={user_message}"


class FakeSpaceXClient(SpaceXClientInterface):
    """No-op client for API tests."""

    async def get_latest_launch(self) -> LaunchRecord:
        return {}

    async def get_next_launch(self) -> LaunchRecord:
        return {}

    async def get_launches(
        self,
        *,
        year: int | None = None,
        successful: bool | None = None,
        limit: int = 100,
    ) -> Sequence[LaunchRecord]:
        del year, successful, limit
        return []

    async def search_launches(self, query: str, *, limit: int = 10) -> Sequence[LaunchRecord]:
        del query, limit
        return []

    async def get_rocket(self, rocket_id: str) -> LaunchRecord:
        del rocket_id
        return {}

    async def get_successful_launches_by_rocket(
        self, rocket_name: str, *, limit: int = 10
    ) -> Sequence[LaunchRecord]:
        del rocket_name, limit
        return []

    async def query_launches_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 10,
        populate_rocket: bool = True,
        populate_launchpad: bool = False,
        sort_direction: Literal["asc", "desc"] = "desc",
    ) -> QueryResponse:
        del query, limit, populate_rocket, populate_launchpad, sort_direction
        return {"docs": [], "totalDocs": 0}

    async def query_rockets_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 10,
    ) -> QueryResponse:
        del query, limit
        return {"docs": [], "totalDocs": 0}

    async def close(self) -> None:
        return None


def _build_test_container() -> AppContainer:
    settings = Settings(
        openai_api_key="test-openai-key",
        langchain_api_key="test-langsmith-key",
        langchain_tracing_v2=False,
        openai_model="gpt-4.1-mini",
        spacex_api_base_url="https://api.spacexdata.com/v5",
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
