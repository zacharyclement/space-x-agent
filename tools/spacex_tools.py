"""LangChain tools and core tool logic for SpaceX lookups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain.tools import tool
from langchain_core.tools import BaseTool

from services.spacex_client_interface import LaunchRecord, SpaceXClientInterface


def _string_value(value: object, *, default: str = "unknown") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _format_launch_line(launch: Mapping[str, object]) -> str:
    name = _string_value(launch.get("name"))
    date_utc = _string_value(launch.get("date_utc"))
    success = launch.get("success")
    if success is True:
        status = "successful"
    elif success is False:
        status = "unsuccessful"
    else:
        status = "outcome unknown"

    rocket_name = "unknown rocket"
    rocket = launch.get("rocket")
    if isinstance(rocket, Mapping):
        rocket_name = _string_value(rocket.get("name"), default=rocket_name)

    return f"{name} | {date_utc} | {rocket_name} | {status}"


async def latest_launch_tool_logic(client: SpaceXClientInterface) -> str:
    """Build a concise summary of the latest launch.

    Args:
        client: Injected SpaceX client.

    Returns:
        Human-readable summary text.
    """
    launch = await client.get_latest_launch()
    details = _string_value(launch.get("details"), default="No extra details available.")
    return f"Latest launch: {_format_launch_line(launch)}. Details: {details}"


async def next_launch_tool_logic(client: SpaceXClientInterface) -> str:
    """Build a concise summary of the next launch."""
    launch = await client.get_next_launch()
    return f"Next launch: {_format_launch_line(launch)}."


async def count_successful_launches_tool_logic(client: SpaceXClientInterface, year: int) -> str:
    """Count successful launches for a year."""
    launches = await client.get_launches(year=year, successful=True, limit=300)
    return f"SpaceX completed {len(launches)} successful launches in {year}."


async def search_launches_tool_logic(
    client: SpaceXClientInterface, query: str, *, limit: int = 5
) -> str:
    """Search launches by query and return a compact list."""
    launches = await client.search_launches(query, limit=limit)
    return _format_search_results(launches, query=query)


async def mission_rocket_tool_logic(client: SpaceXClientInterface, mission_name: str) -> str:
    """Find which rocket was used for a mission."""
    launches = await client.search_launches(mission_name, limit=5)
    if not launches:
        return f"No launch found matching '{mission_name}'."

    best_match = launches[0]
    rocket = best_match.get("rocket")
    rocket_name = "unknown rocket"

    if isinstance(rocket, Mapping):
        rocket_name = _string_value(rocket.get("name"), default=rocket_name)
    elif isinstance(rocket, str):
        rocket_data = await client.get_rocket(rocket)
        rocket_name = _string_value(rocket_data.get("name"), default=rocket_name)

    launch_name = _string_value(best_match.get("name"))
    return f"The mission '{launch_name}' used rocket '{rocket_name}'."


async def successful_launches_for_rocket_tool_logic(
    client: SpaceXClientInterface, rocket_name: str, *, limit: int = 10
) -> str:
    """List successful launches for a given rocket."""
    launches = await client.get_successful_launches_by_rocket(rocket_name, limit=limit)
    if not launches:
        return f"No successful launches found for rocket '{rocket_name}'."

    lines = [_format_launch_line(launch) for launch in launches]
    return f"Successful launches for '{rocket_name}' (showing up to {limit}):\n" + "\n".join(
        f"- {line}" for line in lines
    )


def create_spacex_tools(client: SpaceXClientInterface) -> list[BaseTool]:
    """Create SpaceX tools with dependency-injected client.

    Args:
        client: SpaceX API client implementation.

    Returns:
        Tool list to register with the agent.
    """

    @tool("get_latest_launch")
    async def get_latest_launch() -> str:
        """Return the most recent SpaceX launch with key details."""
        return await latest_launch_tool_logic(client)

    @tool("get_next_launch")
    async def get_next_launch() -> str:
        """Return the next scheduled SpaceX launch."""
        return await next_launch_tool_logic(client)

    @tool("count_successful_launches")
    async def count_successful_launches(year: int) -> str:
        """Count successful SpaceX launches for a specific year."""
        return await count_successful_launches_tool_logic(client, year)

    @tool("search_launches")
    async def search_launches(query: str, limit: int = 5) -> str:
        """Search launches by mission name and return concise results."""
        return await search_launches_tool_logic(client, query, limit=limit)

    @tool("mission_rocket")
    async def mission_rocket(mission_name: str) -> str:
        """Return the rocket used for a mission."""
        return await mission_rocket_tool_logic(client, mission_name)

    @tool("successful_launches_for_rocket")
    async def successful_launches_for_rocket(rocket_name: str, limit: int = 10) -> str:
        """List successful launches for a rocket model."""
        return await successful_launches_for_rocket_tool_logic(client, rocket_name, limit=limit)

    return [
        get_latest_launch,
        get_next_launch,
        count_successful_launches,
        search_launches,
        mission_rocket,
        successful_launches_for_rocket,
    ]


def _format_search_results(launches: Sequence[LaunchRecord], *, query: str) -> str:
    if not launches:
        return f"No launches found for query '{query}'."
    lines = [_format_launch_line(launch) for launch in launches]
    return "Matching launches:\n" + "\n".join(f"- {line}" for line in lines)
