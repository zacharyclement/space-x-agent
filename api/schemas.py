"""Request and response schemas for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Incoming chat payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Max length is enforced in the route using settings.max_user_message_chars
    # so environment configuration remains the single source of truth.
    message: str = Field(..., min_length=1)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    """Chat response payload."""

    thread_id: str
    reply: str
