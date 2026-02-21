"""Tests for SpaceX tool logic."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import pytest

from services.spacex_client_interface import LaunchRecord, QueryResponse, SpaceXClientInterface
from tools.spacex_tools import (
    count_successful_launches_tool_logic,
    latest_launch_tool_logic,
    mission_rocket_tool_logic,
    next_launch_tool_logic,
    search_launches_tool_logic,
    successful_launches_for_rocket_tool_logic,
)


class MockSpaceXClient(SpaceXClientInterface):
    """Deterministic in-memory implementation for tests."""

    async def get_latest_launch(self) -> LaunchRecord:
        return {
            "name": "Starlink 9-1",
            "date_utc": "2025-08-10T11:00:00.000Z",
            "success": True,
            "details": "Deployment mission.",
            "rocket": {"name": "Falcon 9"},
        }

    async def get_next_launch(self) -> LaunchRecord:
        return {
            "name": "CRS-35",
            "date_utc": "2025-10-02T12:00:00.000Z",
            "success": None,
            "rocket": {"name": "Falcon 9"},
            "launchpad": {
                "full_name": "Kennedy Space Center Historic Launch Complex 39A",
                "locality": "Cape Canaveral",
                "region": "Florida",
            },
        }

    async def get_launches(
        self,
        *,
        year: int | None = None,
        successful: bool | None = None,
        limit: int = 100,
    ) -> Sequence[LaunchRecord]:
        del year, successful
        launches: list[LaunchRecord] = [
            {"name": f"Launch {index}", "date_utc": "2024-01-01T00:00:00.000Z", "success": True}
            for index in range(5)
        ]
        return launches[:limit]

    async def search_launches(self, query: str, *, limit: int = 10) -> Sequence[LaunchRecord]:
        launch = {
            "name": "Starlink 9-1",
            "date_utc": "2025-08-10T11:00:00.000Z",
            "success": True,
            "rocket": {"name": "Falcon 9"},
        }
        if "missing" in query.lower():
            return []
        return [launch][:limit]

    async def get_rocket(self, rocket_id: str) -> LaunchRecord:
        del rocket_id
        return {"name": "Falcon 9"}

    async def get_successful_launches_by_rocket(
        self, rocket_name: str, *, limit: int = 10
    ) -> Sequence[LaunchRecord]:
        if rocket_name.lower() != "falcon 9":
            return []
        launches: list[LaunchRecord] = [
            {
                "name": f"Falcon Mission {index}",
                "date_utc": f"2024-0{index}-01T00:00:00.000Z",
                "success": True,
                "rocket": {"name": "Falcon 9"},
            }
            for index in range(1, 4)
        ]
        return cast(Sequence[LaunchRecord], launches[:limit])

    async def close(self) -> None:
        return None

    async def query_launches_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 10,
        populate_rocket: bool = True,
        populate_launchpad: bool = False,
        sort_direction: Literal["asc", "desc"] = "desc",
    ) -> QueryResponse:
        del populate_rocket, populate_launchpad, sort_direction

        latest = await self.get_latest_launch()
        next_launch = await self.get_next_launch()

        if query.get("upcoming") is True:
            docs = [next_launch]
            return {"docs": docs[:limit], "totalDocs": len(docs)}

        if query.get("upcoming") is False:
            docs = [latest]
            return {"docs": docs[:limit], "totalDocs": len(docs)}

        name_query = query.get("name")
        if isinstance(name_query, dict):
            regex = str(name_query.get("$regex", "")).lower()
            if "missing" in regex:
                docs = []
            elif "starlink" in regex:
                docs = [latest]
            else:
                docs = []
            return {"docs": docs[:limit], "totalDocs": len(docs)}

        if query.get("success") is True and "date_utc" in query:
            launches = await self.get_launches(year=2024, successful=True, limit=300)
            docs = list(launches)
            return {"docs": docs[:limit], "totalDocs": len(docs)}

        if query.get("rocket") == "rocket-f9" and query.get("success") is True:
            launches = await self.get_successful_launches_by_rocket("Falcon 9", limit=limit)
            docs = list(launches)
            return {"docs": docs[:limit], "totalDocs": len(docs)}

        docs = [latest, next_launch]
        return {"docs": docs[:limit], "totalDocs": len(docs)}

    async def query_rockets_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 10,
    ) -> QueryResponse:
        name_query = query.get("name")
        if isinstance(name_query, dict) and "falcon" in str(name_query.get("$regex", "")).lower():
            docs = [{"id": "rocket-f9", "name": "Falcon 9"}]
            return {"docs": docs[:limit], "totalDocs": len(docs)}
        return {"docs": [], "totalDocs": 0}


@pytest.mark.asyncio
async def test_latest_launch_tool_logic_returns_raw_payload() -> None:
    client = MockSpaceXClient()
    result = await latest_launch_tool_logic(client)
    payload = json.loads(result)
    assert payload["totalDocs"] == 1
    assert payload["docs"][0]["name"] == "Starlink 9-1"


@pytest.mark.asyncio
async def test_count_successful_launches_tool_logic() -> None:
    client = MockSpaceXClient()
    result = await count_successful_launches_tool_logic(client, 2024)
    payload = json.loads(result)
    assert payload["totalDocs"] == 5


@pytest.mark.asyncio
async def test_next_launch_tool_logic_returns_raw_payload() -> None:
    client = MockSpaceXClient()
    result = await next_launch_tool_logic(client)
    payload = json.loads(result)
    assert payload["docs"][0]["name"] == "CRS-35"
    assert payload["docs"][0]["launchpad"]["locality"] == "Cape Canaveral"


@pytest.mark.asyncio
async def test_mission_rocket_tool_logic() -> None:
    client = MockSpaceXClient()
    result = await mission_rocket_tool_logic(client, "Starlink 9-1")
    payload = json.loads(result)
    assert payload["mission_name"] == "Starlink 9-1"
    assert payload["filtered_response"]["totalDocs"] >= 1


@pytest.mark.asyncio
async def test_search_launches_no_match() -> None:
    client = MockSpaceXClient()
    result = await search_launches_tool_logic(client, "missing")
    payload = json.loads(result)
    assert payload["totalDocs"] == 0


@pytest.mark.asyncio
async def test_successful_launches_for_rocket() -> None:
    client = MockSpaceXClient()
    result = await successful_launches_for_rocket_tool_logic(client, "Falcon 9", limit=2)
    payload = json.loads(result)
    assert payload["rocket_id"] == "rocket-f9"
    assert payload["successful_launches_response"]["docs"][0]["name"] == "Falcon Mission 1"
