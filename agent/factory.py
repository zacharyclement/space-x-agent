"""Factory and runner for LangChain SpaceX agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from agent.middleware import build_trim_messages_middleware, handle_spacex_tool_errors
from agent.state import ChatAgentState
from core.exceptions import SpaceXAgentError
from core.settings import Settings
from services.spacex_client_interface import SpaceXClientInterface
from tools.spacex_tools import create_spacex_tools

SYSTEM_PROMPT = """
You are a SpaceX assistant powered by the r/SpaceX community API.

IMPORTANT — DATA COVERAGE:
The SpaceX API database is frozen at approximately October 2022. The most recent
past launch on record is Crew-5 (2022-10-05). There is no data for launches after
that date, including 2023, 2024, or 2025. When a user asks about events after
October 2022, clearly state that the data is not available in the API and explain
the limitation — do not guess or fabricate results.

TOOL USAGE:
- Always use tools for factual SpaceX claims; never fabricate data.
- Tool outputs are raw SpaceX API JSON. Parse them carefully before answering.
- If totalDocs is 0 or docs is empty, the record does not exist in the database.
- If a user query is ambiguous, ask a concise clarifying question.
- If a tool fails, explain what happened and suggest a retry.
""".strip()


class SupportsAsyncInvoke(Protocol):
    """Protocol for LangChain agent async invocation behavior."""

    async def ainvoke(
        self,
        input_payload: Mapping[str, object],
        config: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Invoke agent asynchronously with payload and optional config."""


class AgentRunnerInterface(Protocol):
    """Abstraction used by the API layer."""

    async def run(self, *, user_message: str, thread_id: str) -> str:
        """Run one turn of the conversation."""


class LangChainAgentRunner(AgentRunnerInterface):
    """Adapter that executes a LangChain agent and extracts text output."""

    def __init__(self, agent: SupportsAsyncInvoke) -> None:
        """Initialize with an agent instance."""
        self._agent = agent

    async def run(self, *, user_message: str, thread_id: str) -> str:
        """Invoke the agent while preserving short-term memory by thread id.

        Args:
            user_message: The latest user message.
            thread_id: Memory thread identifier.

        Returns:
            Assistant response text.

        Raises:
            SpaceXAgentError: If no assistant response can be extracted.
        """
        payload = {"messages": [{"role": "user", "content": user_message}]}
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._agent.ainvoke(payload, config)
        response_text = self._extract_last_ai_message(result)
        if not response_text:
            raise SpaceXAgentError("Agent returned no assistant response.")
        return response_text

    def _extract_last_ai_message(self, result: Mapping[str, object]) -> str:
        messages = result.get("messages")
        if not isinstance(messages, Sequence):
            raise SpaceXAgentError("Agent output did not include a messages list.")

        for message in reversed(messages):
            message_type = _message_type(message)
            if message_type not in {"ai", "assistant"}:
                continue
            content = _message_content(message)
            if content:
                return content
        raise SpaceXAgentError("Agent output did not include an assistant message.")


def build_agent_runner(
    *,
    settings: Settings,
    spacex_client: SpaceXClientInterface,
) -> AgentRunnerInterface:
    """Create a configured agent runner with short-term memory.

    Args:
        settings: Runtime settings.
        spacex_client: Dependency-injected SpaceX client.

    Returns:
        Ready-to-use agent runner.
    """
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    tools = create_spacex_tools(spacex_client)
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            build_trim_messages_middleware(settings.max_history_messages),
            handle_spacex_tool_errors,
        ],
        state_schema=ChatAgentState,
        checkpointer=InMemorySaver(),
    )
    return LangChainAgentRunner(agent=agent)


def _message_type(message: object) -> str:
    if isinstance(message, Mapping):
        raw = message.get("type", message.get("role", ""))
        return str(raw).lower()
    message_type = getattr(message, "type", "")
    if isinstance(message_type, str):
        return message_type.lower()
    return ""


def _message_content(message: object) -> str:
    if isinstance(message, Mapping):
        raw = message.get("content", "")
        return _coerce_content(raw)
    raw = getattr(message, "content", "")
    return _coerce_content(raw)


def _coerce_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    parts.append(stripped)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts).strip()
    return ""
