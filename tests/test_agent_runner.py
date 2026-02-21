"""Tests for LangChain agent runner behavior."""

from __future__ import annotations

import pytest

from agent.factory import LangChainAgentRunner
from core.exceptions import SpaceXAgentError


class FakeAsyncAgent:
    """Simple async-only agent stub for runner tests."""

    async def ainvoke(
        self,
        input_payload: dict[str, object],
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del input_payload, config
        return {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "next launch is from KSC"},
            ]
        }


class EmptyResponseAsyncAgent:
    """Agent stub that returns no assistant message."""

    async def ainvoke(
        self,
        input_payload: dict[str, object],
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del input_payload, config
        return {"messages": [{"role": "user", "content": "hello"}]}


@pytest.mark.asyncio
async def test_runner_uses_async_invoke_and_extracts_reply() -> None:
    runner = LangChainAgentRunner(agent=FakeAsyncAgent())
    reply = await runner.run(user_message="What is the next launch?", thread_id="thread-1")

    assert reply == "next launch is from KSC"


@pytest.mark.asyncio
async def test_runner_raises_when_no_assistant_message() -> None:
    runner = LangChainAgentRunner(agent=EmptyResponseAsyncAgent())

    with pytest.raises(SpaceXAgentError, match="assistant message"):
        await runner.run(user_message="What is the next launch?", thread_id="thread-2")
