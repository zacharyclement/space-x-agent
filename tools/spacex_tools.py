"""LangChain tools and core tool logic for SpaceX lookups."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from langchain.tools import tool
from langchain_core.tools import BaseTool

from core.logging import get_logger
from services.spacex_client_interface import QueryResponse, SpaceXClientInterface

logger = get_logger(__name__)


def _to_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _docs_count(payload: Mapping[str, object]) -> int:
    docs = payload.get("docs")
    if not isinstance(docs, Sequence):
        return 0
    return len(docs)


def _first_doc_id(payload: Mapping[str, object]) -> str:
    docs = payload.get("docs")
    if not isinstance(docs, Sequence) or not docs:
        return ""
    first = docs[0]
    if not isinstance(first, Mapping):
        return ""
    record_id = first.get("id")
    if not isinstance(record_id, str):
        return ""
    return record_id


async def latest_launch_tool_logic(client: SpaceXClientInterface) -> str:
    """Return raw API payload for the latest launch query."""
    logger.debug("tool_latest_launch_started")
    response = await client.query_launches_raw(
        query={"upcoming": False},
        limit=1,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
    )
    logger.debug("tool_latest_launch_succeeded", docs_count=_docs_count(response))
    return _to_json(response)


async def next_launch_tool_logic(client: SpaceXClientInterface) -> str:
    """Return raw API payload for the next launch query."""
    logger.debug("tool_next_launch_started")
    response = await client.query_launches_raw(
        query={"upcoming": True},
        limit=1,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="asc",
    )
    logger.debug("tool_next_launch_succeeded", docs_count=_docs_count(response))
    return _to_json(response)


async def count_successful_launches_tool_logic(client: SpaceXClientInterface, year: int) -> str:
    """Return raw API payload for successful launches in a year."""
    logger.debug("tool_count_successful_launches_started", year=year)
    response = await client.query_launches_raw(
        query={
            "date_utc": {
                "$gte": f"{year}-01-01T00:00:00.000Z",
                "$lt": f"{year + 1}-01-01T00:00:00.000Z",
            },
            "success": True,
        },
        limit=300,
        populate_rocket=True,
        sort_direction="desc",
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
    """Return raw API payload for a launch-name regex search."""
    logger.debug("tool_search_launches_started", query=query, limit=limit)
    response = await client.query_launches_raw(
        query={"name": {"$regex": query, "$options": "i"}},
        limit=limit,
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
    """Return raw payloads for mission lookup without tool-side matching."""
    logger.debug("tool_mission_rocket_started", mission_name=mission_name)
    filtered_response = await client.query_launches_raw(
        query={"name": {"$regex": mission_name, "$options": "i"}},
        limit=25,
        populate_rocket=True,
        populate_launchpad=True,
    )
    recent_response = await client.query_launches_raw(
        query={"upcoming": False},
        limit=100,
        populate_rocket=True,
        populate_launchpad=True,
        sort_direction="desc",
    )
    result: dict[str, object] = {
        "mission_name": mission_name,
        "filtered_response": filtered_response,
        "recent_response": recent_response,
    }
    logger.debug(
        "tool_mission_rocket_succeeded",
        mission_name=mission_name,
        filtered_docs=_docs_count(filtered_response),
        recent_docs=_docs_count(recent_response),
    )
    return _to_json(result)


async def successful_launches_for_rocket_tool_logic(
    client: SpaceXClientInterface, rocket_name: str, *, limit: int = 10
) -> str:
    """Return raw payloads for rocket lookup and successful-launch query."""
    logger.debug(
        "tool_successful_launches_for_rocket_started",
        rocket_name=rocket_name,
        limit=limit,
    )
    rocket_response = await client.query_rockets_raw(
        query={"name": {"$regex": rocket_name, "$options": "i"}},
        limit=1,
    )
    rocket_id = _first_doc_id(rocket_response)
    launches_response: QueryResponse = {
        "docs": [],
        "totalDocs": 0,
    }
    if rocket_id:
        launches_response = await client.query_launches_raw(
            query={"rocket": rocket_id, "success": True},
            limit=limit,
            populate_rocket=True,
            sort_direction="desc",
        )
    result: dict[str, object] = {
        "rocket_name": rocket_name,
        "rocket_response": rocket_response,
        "rocket_id": rocket_id,
        "successful_launches_response": launches_response,
    }
    logger.debug(
        "tool_successful_launches_for_rocket_succeeded",
        rocket_name=rocket_name,
        rocket_docs=_docs_count(rocket_response),
        launch_docs=_docs_count(launches_response),
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
        """Return raw SpaceX API response for latest launch query."""
        return await latest_launch_tool_logic(client)

    @tool("get_next_launch")
    async def get_next_launch() -> str:
        """Return raw SpaceX API response for next launch query."""
        return await next_launch_tool_logic(client)

    @tool("count_successful_launches")
    async def count_successful_launches(year: int) -> str:
        """Return raw SpaceX API response for successful launches in the given year."""
        return await count_successful_launches_tool_logic(client, year)

    @tool("search_launches")
    async def search_launches(query: str, limit: int = 25) -> str:
        """Return raw SpaceX API response for launch-name search."""
        return await search_launches_tool_logic(client, query, limit=limit)

    @tool("mission_rocket")
    async def mission_rocket(mission_name: str) -> str:
        """Return raw payloads to help identify a mission's rocket."""
        return await mission_rocket_tool_logic(client, mission_name)

    @tool("successful_launches_for_rocket")
    async def successful_launches_for_rocket(rocket_name: str, limit: int = 10) -> str:
        """Return raw payloads for successful launches by rocket."""
        return await successful_launches_for_rocket_tool_logic(client, rocket_name, limit=limit)

    return [
        get_latest_launch,
        get_next_launch,
        count_successful_launches,
        search_launches,
        mission_rocket,
        successful_launches_for_rocket,
    ]
