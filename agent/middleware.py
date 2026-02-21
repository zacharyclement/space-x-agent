"""Agent middleware for short-term memory and graceful tool errors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import before_model, wrap_tool_call
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from core.exceptions import SpaceXApiError


def build_trim_messages_middleware(
    max_messages: int,
) -> Callable[[AgentState, Runtime], dict[str, Any] | None]:
    """Create middleware that keeps message history bounded.

    Args:
        max_messages: Maximum number of messages to keep in state.

    Returns:
        A `before_model` middleware function.
    """

    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        del runtime  # Runtime is currently unused but part of middleware signature.
        messages = state["messages"]
        if max_messages < 2 or len(messages) <= max_messages:
            return None

        first_message = messages[0]
        recent_messages = list(messages[-(max_messages - 1) :])
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                first_message,
                *recent_messages,
            ]
        }

    return trim_messages


@wrap_tool_call
async def handle_spacex_tool_errors(
    request: object,
    handler: Callable[[object], Awaitable[object]],
) -> object:
    """Convert SpaceX API errors into tool messages the model can recover from."""
    try:
        return await handler(request)
    except SpaceXApiError as exc:
        tool_call_id = ""
        tool_call = getattr(request, "tool_call", None)
        if isinstance(tool_call, Mapping):
            tool_call_id = str(tool_call.get("id", ""))
        return ToolMessage(
            content=(
                "SpaceX API request failed. "
                f"Error: {exc}. Ask for a narrower query or try again shortly."
            ),
            tool_call_id=tool_call_id,
        )
