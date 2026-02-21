"""Agent state models."""

from __future__ import annotations

from langchain.agents import AgentState
from pydantic import Field


class ChatAgentState(AgentState):
    """Custom state schema for short-term memory fields."""

    user_id: str | None = None
    preferences: dict[str, str] = Field(default_factory=dict)
