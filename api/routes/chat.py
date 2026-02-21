"""Chat API routes."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import AgentRunnerProtocol, get_agent_runner, get_app_settings
from api.schemas import ChatRequest, ChatResponse
from core.exceptions import SpaceXAgentError
from core.logging import get_logger
from core.settings import Settings

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_app_settings),
    agent_runner: AgentRunnerProtocol = Depends(get_agent_runner),
) -> ChatResponse:
    """Handle one user chat turn.

    Args:
        payload: User chat payload.
        settings: Application settings.
        agent_runner: Configured agent runner.

    Returns:
        Chat response with assistant text and thread id.
    """
    message = payload.message.strip()
    if len(message) > settings.max_user_message_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"Message exceeds max length of {settings.max_user_message_chars} characters."),
        )

    thread_id = payload.thread_id or str(uuid4())
    try:
        reply = await agent_runner.run(user_message=message, thread_id=thread_id)
    except SpaceXAgentError as exc:
        logger.error("chat_failed", error=str(exc), thread_id=thread_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to produce a response from the SpaceX agent.",
        ) from exc

    logger.info("chat_succeeded", thread_id=thread_id)
    return ChatResponse(thread_id=thread_id, reply=reply)
