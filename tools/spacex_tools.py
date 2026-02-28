"""LangChain tools for SpaceX data lookups.

Note: The SpaceX community API (api.spacexdata.com) database is frozen at
approximately October 2022. Queries for data beyond that date will return
empty results — this is expected and not a bug.
"""

from __future__ import annotations

import json

from langchain.tools import tool
from langchain_core.tools import BaseTool

from core.logging import get_logger
from services.spacex_client_interface import SpaceXClientInterface

logger = get_logger(__name__)

# The SpaceX API database does not contain data beyond this date.
_API_DATA_CUTOFF_YEAR = 2022
_API_DATA_CUTOFF_DATE = "2022-10-05"


def _to_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


# ---------------------------------------------------------------------------
# Pure tool logic functions (testable without LangChain)
# ---------------------------------------------------------------------------


async def get_latest_launch_logic(client: SpaceXClientInterface) -> str:
    """Fetch the latest SpaceX launch and return it as a JSON string."""
    logger.debug("tool_get_latest_launch_started")
    result = await client.get_latest_launch()
    logger.debug("tool_get_latest_launch_succeeded", name=result.get("name"))
    return _to_json(result)  # noqa: RET504


async def get_next_launch_logic(client: SpaceXClientInterface) -> str:
    """Fetch the next upcoming launch per the API and return it as a JSON string.

    Note: because the API data is frozen at Oct 2022, the "next" launch
    returned will also be from 2022 (the first launch still marked upcoming
    in the stale database).
    """
    logger.debug("tool_get_next_launch_started")
    result = await client.get_next_launch()
    logger.debug("tool_get_next_launch_succeeded", name=result.get("name"))
    return _to_json(result)  # noqa: RET504


async def count_launches_in_year_logic(client: SpaceXClientInterface, year: int) -> str:
    """Query all launches in a given year and return the raw response as a JSON string.

    Returns a metadata envelope with a ``data_available`` flag so the caller
    can easily detect when the year is outside the API's coverage window.
    """
    logger.debug("tool_count_launches_in_year_started", year=year)

    # Short-circuit for years known to be outside the API's data coverage.
    if year > _API_DATA_CUTOFF_YEAR:
        logger.debug("tool_count_launches_in_year_out_of_range", year=year)
        return _to_json(
            {
                "data_available": False,
                "year": year,
                "reason": (
                    f"The SpaceX API database is frozen at {_API_DATA_CUTOFF_DATE}. "
                    f"No launch data is available for {year} or any later year."
                ),
                "docs": [],
                "totalDocs": 0,
            }
        )

    result = await client.query_launches(
        query={
            "date_utc": {
                "$gte": f"{year}-01-01T00:00:00.000Z",
                "$lt": f"{year + 1}-01-01T00:00:00.000Z",
            }
        },
        options={"limit": 200, "sort": {"date_utc": "asc"}},
    )
    docs = result.get("docs")
    returned = len(docs) if isinstance(docs, list) else 0
    logger.debug(
        "tool_count_launches_in_year_succeeded",
        year=year,
        total_docs=result.get("totalDocs"),
        returned_docs=returned,
    )
    # Wrap with a data_available flag for clarity
    return _to_json({"data_available": True, "year": year, **result})


async def search_launches_logic(client: SpaceXClientInterface, name: str) -> str:
    """Search launches by mission name and return the raw response as a JSON string."""
    logger.debug("tool_search_launches_started", name=name)
    result = await client.query_launches(
        query={"name": {"$regex": name, "$options": "i"}},
        options={
            "limit": 50,
            "sort": {"date_utc": "desc"},
            "populate": [
                {"path": "rocket", "select": "name"},
                {"path": "launchpad", "select": "name full_name locality region"},
            ],
        },
    )
    docs = result.get("docs")
    returned = len(docs) if isinstance(docs, list) else 0
    logger.debug("tool_search_launches_succeeded", name=name, returned_docs=returned)
    return _to_json(result)


async def get_launches_by_rocket_logic(
    client: SpaceXClientInterface,
    rocket_name: str,
    *,
    successful_only: bool = False,
) -> str:
    """Fetch launches for a rocket (by name) and return raw results as a JSON string.

    Looks up the rocket ID first via GET /rockets, then queries launches
    filtered by that rocket ID.
    """
    logger.debug(
        "tool_get_launches_by_rocket_started",
        rocket_name=rocket_name,
        successful_only=successful_only,
    )

    rockets = await client.get_rockets()
    matched = [r for r in rockets if rocket_name.lower() in r.get("name", "").lower()]

    if not matched:
        logger.debug("tool_get_launches_by_rocket_no_rocket_found", rocket_name=rocket_name)
        return _to_json(
            {
                "error": f"No rocket found matching '{rocket_name}'.",
                "rockets_available": [r.get("name") for r in rockets],
            }
        )

    rocket_ids = [r["id"] for r in matched if "id" in r]
    launch_query: dict = {"rocket": {"$in": rocket_ids}}
    if successful_only:
        launch_query["success"] = True

    result = await client.query_launches(
        query=launch_query,
        options={
            "limit": 200,
            "sort": {"date_utc": "desc"},
            "populate": [
                {"path": "rocket", "select": "name"},
                {"path": "launchpad", "select": "name full_name locality region"},
            ],
        },
    )
    docs = result.get("docs")
    logger.debug(
        "tool_get_launches_by_rocket_succeeded",
        rocket_name=rocket_name,
        rocket_ids=rocket_ids,
        returned_docs=len(docs) if isinstance(docs, list) else 0,
    )
    return _to_json(result)


async def get_launches_from_location_logic(
    client: SpaceXClientInterface,
    location: str,
) -> str:
    """Fetch launches from a location and return raw results as a JSON string.

    Matches location against launchpad name, full_name, locality, or region
    via GET /launchpads, then queries launches filtered by matching pad IDs.
    """
    logger.debug("tool_get_launches_from_location_started", location=location)

    launchpads = await client.get_launchpads()
    location_lower = location.lower()
    matched = [
        lp
        for lp in launchpads
        if any(
            location_lower in str(lp.get(field, "")).lower()
            for field in ("name", "full_name", "locality", "region")
        )
    ]

    if not matched:
        logger.debug("tool_get_launches_from_location_no_pad_found", location=location)
        return _to_json(
            {
                "error": f"No launchpad found matching '{location}'.",
                "launchpads_available": [lp.get("full_name") for lp in launchpads],
            }
        )

    pad_ids = [lp["id"] for lp in matched if "id" in lp]
    result = await client.query_launches(
        query={"launchpad": {"$in": pad_ids}},
        options={
            "limit": 50,
            "sort": {"date_utc": "desc"},
            "populate": [
                {"path": "rocket", "select": "name"},
                {"path": "launchpad", "select": "name full_name locality region"},
            ],
        },
    )
    docs = result.get("docs")
    logger.debug(
        "tool_get_launches_from_location_succeeded",
        location=location,
        pad_ids=pad_ids,
        returned_docs=len(docs) if isinstance(docs, list) else 0,
    )
    return _to_json(result)


# ---------------------------------------------------------------------------
# LangChain tool factory
# ---------------------------------------------------------------------------


def create_spacex_tools(client: SpaceXClientInterface) -> list[BaseTool]:
    """Create SpaceX LangChain tools with a dependency-injected client.

    Args:
        client: SpaceX API client implementation.

    Returns:
        Tool list to register with the agent.
    """

    @tool("get_latest_launch")
    async def get_latest_launch() -> str:
        """Return the most recent past SpaceX launch as raw JSON.

        Use this tool when the user asks about the last or most recent launch.
        The API database is frozen at October 2022, so the result will be
        the most recent launch known to the API at that time (Crew-5).
        """
        return await get_latest_launch_logic(client)

    @tool("get_next_launch")
    async def get_next_launch() -> str:
        """Return the next upcoming SpaceX launch per the API database as raw JSON.

        Use this tool when the user asks about the next scheduled launch.
        Note: the API data is frozen at October 2022, so the result reflects
        the first launch that was still 'upcoming' at that cutoff date —
        not a real future launch.
        """
        return await get_next_launch_logic(client)

    @tool("count_launches_in_year")
    async def count_launches_in_year(year: int) -> str:
        """Return all SpaceX launches for a given calendar year as raw JSON.

        Use this tool when the user asks how many launches occurred in a year.
        Data is only available for launches up to and including 2022. For
        years after 2022 the response will include data_available=false and
        explain the limitation.

        Args:
            year: The four-digit calendar year (e.g. 2022).
        """
        return await count_launches_in_year_logic(client, year)

    @tool("search_launches")
    async def search_launches(name: str) -> str:
        """Search SpaceX launches by mission name and return matching launches as raw JSON.

        Use this tool when the user asks about a specific mission by name
        (e.g. 'Falcon Heavy', 'Crew-5', 'Starlink 4-36').
        Only missions present in the API database (launches up to Oct 2022)
        will be found. Missions after that date will return empty results.

        Args:
            name: Full or partial mission name to search for.
        """
        return await search_launches_logic(client, name)

    @tool("get_launches_by_rocket")
    async def get_launches_by_rocket(rocket_name: str, successful_only: bool = False) -> str:
        """Return SpaceX launches for a given rocket type as raw JSON.

        Use this tool when the user asks about launches for a specific rocket
        (e.g. 'Falcon 9', 'Falcon Heavy') or wants a success count.
        Results cover launches up to October 2022.

        Args:
            rocket_name: Name or partial name of the rocket (e.g. 'Falcon 9').
            successful_only: If True, only return launches where success is True.
        """
        return await get_launches_by_rocket_logic(
            client, rocket_name, successful_only=successful_only
        )

    @tool("get_launches_from_location")
    async def get_launches_from_location(location: str) -> str:
        """Return SpaceX launches from a given launch location as raw JSON.

        Use this tool when the user asks about launches from a specific place
        such as 'Vandenberg', 'Kennedy Space Center', 'Cape Canaveral', or
        'Boca Chica'. Results cover launches up to October 2022.

        Args:
            location: Location name or partial name to match against launchpad records.
        """
        return await get_launches_from_location_logic(client, location)

    return [
        get_latest_launch,
        get_next_launch,
        count_launches_in_year,
        search_launches,
        get_launches_by_rocket,
        get_launches_from_location,
    ]
