"""Tests for SpaceX tool logic."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

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

    def __init__(self) -> None:
        self.query_launches_raw_calls: list[dict[str, object]] = []
        self.query_rockets_raw_calls: list[dict[str, object]] = []
        self._launch_docs: list[LaunchRecord] = [
            {
                "name": "Starlink 9-1",
                "date_utc": "2025-08-10T11:00:00.000Z",
                "success": True,
                "details": "Deployment mission.",
                "rocket": {"name": "Falcon 9"},
                "upcoming": False,
            },
            {
                "name": "CRS-35",
                "date_utc": "2025-10-02T12:00:00.000Z",
                "success": None,
                "rocket": {"name": "Falcon 9"},
                "launchpad": {
                    "full_name": "Kennedy Space Center Historic Launch Complex 39A",
                    "locality": "Cape Canaveral",
                    "region": "Florida",
                },
                "upcoming": True,
            },
        ]

    async def get_latest_launch(self) -> LaunchRecord:
        return self._launch_docs[0]

    async def get_next_launch(self) -> LaunchRecord:
        return self._launch_docs[1]

    async def get_launches(
        self,
        *,
        year: int | None = None,
        successful: bool | None = None,
        limit: int = 100,
    ) -> Sequence[LaunchRecord]:
        del year, successful
        return self._launch_docs[:limit]

    async def search_launches(self, query: str, *, limit: int = 10) -> Sequence[LaunchRecord]:
        del query
        return self._launch_docs[:limit]

    async def get_rocket(self, rocket_id: str) -> LaunchRecord:
        del rocket_id
        return {"name": "Falcon 9"}

    async def get_successful_launches_by_rocket(
        self, rocket_name: str, *, limit: int = 10
    ) -> Sequence[LaunchRecord]:
        del rocket_name
        return self._launch_docs[:limit]

    async def close(self) -> None:
        return None

    async def query_launches_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 1000,
        populate_rocket: bool = True,
        populate_launchpad: bool = False,
        sort_direction: Literal["asc", "desc"] = "desc",
        select_fields: str | None = None,
    ) -> QueryResponse:
        self.query_launches_raw_calls.append(
            {
                "query": dict(query),
                "limit": limit,
                "populate_rocket": populate_rocket,
                "populate_launchpad": populate_launchpad,
                "sort_direction": sort_direction,
                "select_fields": select_fields,
            }
        )
        docs = self._launch_docs[:limit]
        return {"docs": docs[:limit], "totalDocs": len(docs)}

    async def query_rockets_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 1000,
    ) -> QueryResponse:
        self.query_rockets_raw_calls.append({"query": dict(query), "limit": limit})
        docs: list[LaunchRecord] = [
            {"id": "rocket-f9", "name": "Falcon 9"},
            {"id": "rocket-fh", "name": "Falcon Heavy"},
        ]
        return {"docs": docs[:limit], "totalDocs": len(docs)}


@pytest.mark.asyncio
async def test_latest_launch_tool_logic_returns_raw_payload() -> None:
    client = MockSpaceXClient()
    result = await latest_launch_tool_logic(client)
    payload = json.loads(result)
    assert payload["totalDocs"] == 2
    assert payload["docs"][0]["name"] == "Starlink 9-1"
    assert client.query_launches_raw_calls[-1]["query"] == {}
    assert client.query_launches_raw_calls[-1]["limit"] == 1000


@pytest.mark.asyncio
async def test_count_successful_launches_tool_logic() -> None:
    client = MockSpaceXClient()
    result = await count_successful_launches_tool_logic(client, 2024)
    payload = json.loads(result)
    assert payload["totalDocs"] == 2
    assert client.query_launches_raw_calls[-1]["query"] == {}
    assert client.query_launches_raw_calls[-1]["limit"] == 1000


@pytest.mark.asyncio
async def test_next_launch_tool_logic_returns_raw_payload() -> None:
    client = MockSpaceXClient()
    result = await next_launch_tool_logic(client)
    payload = json.loads(result)
    assert payload["docs"][1]["name"] == "CRS-35"
    assert payload["docs"][1]["launchpad"]["locality"] == "Cape Canaveral"
    assert client.query_launches_raw_calls[-1]["query"] == {}
    assert client.query_launches_raw_calls[-1]["limit"] == 1000


@pytest.mark.asyncio
async def test_mission_rocket_tool_logic() -> None:
    client = MockSpaceXClient()
    result = await mission_rocket_tool_logic(client, "Starlink 9-1")
    payload = json.loads(result)
    assert payload["mission_name"] == "Starlink 9-1"
    assert payload["launches_response"]["totalDocs"] == 2
    assert client.query_launches_raw_calls[-1]["query"] == {}
    assert client.query_launches_raw_calls[-1]["limit"] == 1000


@pytest.mark.asyncio
async def test_search_launches_returns_broad_payload() -> None:
    client = MockSpaceXClient()
    result = await search_launches_tool_logic(client, "missing", limit=3)
    payload = json.loads(result)
    assert payload["totalDocs"] == 2
    assert client.query_launches_raw_calls[-1]["query"] == {}
    assert client.query_launches_raw_calls[-1]["limit"] == 1000


@pytest.mark.asyncio
async def test_successful_launches_for_rocket() -> None:
    client = MockSpaceXClient()
    result = await successful_launches_for_rocket_tool_logic(client, "Falcon 9", limit=2)
    payload = json.loads(result)
    assert payload["requested_limit"] == 2
    assert payload["rocket_response"]["totalDocs"] == 2
    assert payload["launches_response"]["totalDocs"] == 2
    assert client.query_rockets_raw_calls[-1]["query"] == {}
    assert client.query_rockets_raw_calls[-1]["limit"] == 1000
    assert client.query_launches_raw_calls[-1]["query"] == {}
    assert client.query_launches_raw_calls[-1]["limit"] == 1000
