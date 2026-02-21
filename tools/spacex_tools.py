"""LangChain tools and core tool logic for SpaceX lookups."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from langchain.tools import tool
from langchain_core.tools import BaseTool

from core.logging import get_logger
from services.spacex_client_interface import SpaceXClientInterface

logger = get_logger(__name__)


def _to_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _docs_count(payload: Mapping[str, object]) -> int:
    docs = payload.get("docs")
    if not isinstance(docs, Sequence):
        return 0
    return len(docs)


async def latest_launch_tool_logic(client: SpaceXClientInterface) -> str:
    """Return broad raw launch payload for latest-launch reasoning."""
    logger.debug("tool_latest_launch_started")
    response = await client.query_launches_raw(
        query={},
        limit=1000,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
    )
    logger.debug("tool_latest_launch_succeeded", docs_count=_docs_count(response))
    return _to_json(response)


async def next_launch_tool_logic(client: SpaceXClientInterface) -> str:
    """Return broad raw launch payload for next-launch reasoning."""
    logger.debug("tool_next_launch_started")
    response = await client.query_launches_raw(
        query={},
        limit=1000,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
    )
    logger.debug("tool_next_launch_succeeded", docs_count=_docs_count(response))
    return _to_json(response)


async def count_successful_launches_tool_logic(client: SpaceXClientInterface, year: int) -> str:
    """Return broad raw launch payload for year/count reasoning."""
    logger.debug("tool_count_successful_launches_started", year=year)
    response = await client.query_launches_raw(
        query={},
        limit=1000,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
        select_fields=None,
    )
    logger.debug(
        "tool_count_successful_launches_succeeded",
        year=year,
        docs_count=_docs_count(response),
    )
    return _to_json(response)


async def search_launches_tool_logic(
    client: SpaceXClientInterface, query: str, *, limit: int = 25
) -> str:
    """Return broad raw launch payload for launch-search reasoning."""
    logger.debug("tool_search_launches_started", query=query, limit=limit)
    response = await client.query_launches_raw(
        query={},
        limit=1000,
        populate_rocket=True,
        populate_launchpad=True,
    )
    logger.debug(
        "tool_search_launches_succeeded",
        query=query,
        limit=limit,
        docs_count=_docs_count(response),
    )
    return _to_json(response)


async def mission_rocket_tool_logic(client: SpaceXClientInterface, mission_name: str) -> str:
    """Return broad raw payload for mission-to-rocket reasoning."""
    logger.debug("tool_mission_rocket_started", mission_name=mission_name)
    launches_response = await client.query_launches_raw(
        query={},
        limit=1000,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
    )
    result: dict[str, object] = {
        "mission_name": mission_name,
        "launches_response": launches_response,
    }
    logger.debug(
        "tool_mission_rocket_succeeded",
        mission_name=mission_name,
        launches_docs=_docs_count(launches_response),
    )
    return _to_json(result)


async def successful_launches_for_rocket_tool_logic(
    client: SpaceXClientInterface, rocket_name: str, *, limit: int = 10
) -> str:
    """Return broad raw rocket and launch payloads for rocket reasoning."""
    logger.debug(
        "tool_successful_launches_for_rocket_started",
        rocket_name=rocket_name,
        limit=limit,
    )
    rocket_response = await client.query_rockets_raw(query={}, limit=1000)
    launches_response = await client.query_launches_raw(
        query={},
        limit=1000,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
    )
    result: dict[str, object] = {
        "rocket_name": rocket_name,
        "requested_limit": limit,
        "rocket_response": rocket_response,
        "launches_response": launches_response,
    }
    logger.debug(
        "tool_successful_launches_for_rocket_succeeded",
        rocket_name=rocket_name,
        rocket_docs=_docs_count(rocket_response),
        launches_docs=_docs_count(launches_response),
    )
    return _to_json(result)


def create_spacex_tools(client: SpaceXClientInterface) -> list[BaseTool]:
    """Create SpaceX tools with dependency-injected client.

    Args:
        client: SpaceX API client implementation.

    Returns:
        Tool list to register with the agent.
    """

    @tool("get_latest_launch")
    async def get_latest_launch() -> str:
        """Return broad raw SpaceX launch payload for latest-launch reasoning."""
        return await latest_launch_tool_logic(client)

    @tool("get_next_launch")
    async def get_next_launch() -> str:
        """Return broad raw SpaceX launch payload for next-launch reasoning."""
        return await next_launch_tool_logic(client)

    @tool("count_successful_launches")
    async def count_successful_launches(year: int) -> str:
        """Return broad raw SpaceX launch payload for year/count reasoning."""
        return await count_successful_launches_tool_logic(client, year)

    @tool("search_launches")
    async def search_launches(query: str, limit: int = 25) -> str:
        """Return broad raw SpaceX launch payload for launch-search reasoning."""
        return await search_launches_tool_logic(client, query, limit=limit)

    @tool("mission_rocket")
    async def mission_rocket(mission_name: str) -> str:
        """Return broad raw SpaceX payload to help identify a mission's rocket."""
        return await mission_rocket_tool_logic(client, mission_name)

    @tool("successful_launches_for_rocket")
    async def successful_launches_for_rocket(rocket_name: str, limit: int = 10) -> str:
        """Return broad raw SpaceX payload for rocket-specific launch reasoning."""
        return await successful_launches_for_rocket_tool_logic(client, rocket_name, limit=limit)

    return [
        get_latest_launch,
        get_next_launch,
        count_successful_launches,
        search_launches,
        mission_rocket,
        successful_launches_for_rocket,
    ]
