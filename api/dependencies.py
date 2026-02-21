"""FastAPI dependency providers and app container."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, cast

from fastapi import Depends, Request

from agent.factory import build_agent_runner
from core.logging import configure_logging
from core.settings import Settings, get_settings
from services.http_spacex_client import HttpSpaceXClient
from services.spacex_client_interface import SpaceXClientInterface


class AgentRunnerProtocol(Protocol):
    """Protocol for chat agent invocation used by the transport layer."""

    async def run(self, *, user_message: str, thread_id: str) -> str:
        """Run one conversational turn."""


@dataclass(slots=True)
class AppContainer:
    """Runtime dependency container."""

    settings: Settings
    spacex_client: SpaceXClientInterface
    agent_runner: AgentRunnerProtocol


def build_app_container() -> AppContainer:
    """Construct and wire concrete dependencies."""

    settings = get_settings()
    configure_logging(settings.log_level)
    _configure_langsmith_environment(settings)

    client = HttpSpaceXClient(
        base_url=str(settings.spacex_api_base_url),
        timeout_seconds=settings.request_timeout_seconds,
    )
    runner = build_agent_runner(settings=settings, spacex_client=client)
    return AppContainer(settings=settings, spacex_client=client, agent_runner=runner)


async def close_app_container(container: AppContainer) -> None:
    """Close resources held by the container."""
    await container.spacex_client.close()


def get_container(request: Request) -> AppContainer:
    """Resolve app container from request state."""
    return cast(AppContainer, request.app.state.container)


def get_app_settings(container: AppContainer = Depends(get_container)) -> Settings:
    """Resolve settings dependency."""
    return container.settings


def get_agent_runner(container: AppContainer = Depends(get_container)) -> AgentRunnerProtocol:
    """Resolve agent runner dependency."""
    return container.agent_runner


def _configure_langsmith_environment(settings: Settings) -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
