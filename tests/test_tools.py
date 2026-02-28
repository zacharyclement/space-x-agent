"""Tests for SpaceX tool logic functions."""

from __future__ import annotations

import json

import pytest

from services.spacex_client_interface import SpaceXClientInterface
from tools.spacex_tools import (
    count_launches_in_year_logic,
    get_latest_launch_logic,
    get_launches_by_rocket_logic,
    get_launches_from_location_logic,
    get_next_launch_logic,
    search_launches_logic,
)


class MockSpaceXClient(SpaceXClientInterface):
    """Deterministic in-memory SpaceX client for unit tests."""

    _FALCON_9_ID = "rocket-f9"
    _FALCON_HEAVY_ID = "rocket-fh"
    _VANDENBERG_ID = "pad-vafb"
    _KSC_ID = "pad-ksc"

    def __init__(self) -> None:
        self.query_launches_calls: list[dict] = []

        self._rockets = [
            {"id": self._FALCON_9_ID, "name": "Falcon 9"},
            {"id": self._FALCON_HEAVY_ID, "name": "Falcon Heavy"},
        ]
        self._launchpads = [
            {
                "id": self._VANDENBERG_ID,
                "name": "VAFB SLC-4E",
                "full_name": "Vandenberg Space Force Base Space Launch Complex 4E",
                "locality": "Vandenberg Space Force Base",
                "region": "California",
            },
            {
                "id": self._KSC_ID,
                "name": "KSC LC 39A",
                "full_name": "Kennedy Space Center Historic Launch Complex 39A",
                "locality": "Cape Canaveral",
                "region": "Florida",
            },
        ]
        self._launches = [
            {
                "id": "launch-1",
                "name": "Starlink 9-1",
                "date_utc": "2024-08-10T11:00:00.000Z",
                "success": True,
                "rocket": {"id": self._FALCON_9_ID, "name": "Falcon 9"},
                "launchpad": {"id": self._VANDENBERG_ID, "locality": "Vandenberg Space Force Base"},
                "upcoming": False,
            },
            {
                "id": "launch-2",
                "name": "Falcon Heavy Demo",
                "date_utc": "2018-02-06T20:45:00.000Z",
                "success": True,
                "rocket": {"id": self._FALCON_HEAVY_ID, "name": "Falcon Heavy"},
                "launchpad": {"id": self._KSC_ID, "locality": "Cape Canaveral"},
                "upcoming": False,
            },
            {
                "id": "launch-3",
                "name": "CRS-35",
                "date_utc": "2025-10-02T12:00:00.000Z",
                "success": None,
                "rocket": {"id": self._FALCON_9_ID, "name": "Falcon 9"},
                "launchpad": {"id": self._KSC_ID, "locality": "Cape Canaveral"},
                "upcoming": True,
            },
        ]

    async def get_latest_launch(self) -> dict:
        # Return the most recent past launch
        return self._launches[0]

    async def get_next_launch(self) -> dict:
        # Return the upcoming launch
        return self._launches[2]

    async def query_launches(self, query: dict, *, options: dict | None = None) -> dict:
        self.query_launches_calls.append({"query": query, "options": options})
        return {"docs": list(self._launches), "totalDocs": len(self._launches)}

    async def get_rockets(self) -> list[dict]:
        return list(self._rockets)

    async def get_launchpads(self) -> list[dict]:
        return list(self._launchpads)

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_launch_logic_returns_single_record() -> None:
    client = MockSpaceXClient()
    result = json.loads(await get_latest_launch_logic(client))
    assert result["name"] == "Starlink 9-1"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_get_next_launch_logic_returns_upcoming_record() -> None:
    client = MockSpaceXClient()
    result = json.loads(await get_next_launch_logic(client))
    assert result["name"] == "CRS-35"
    assert result["upcoming"] is True


@pytest.mark.asyncio
async def test_count_launches_in_year_logic_queries_date_range() -> None:
    client = MockSpaceXClient()
    # 2022 is within the data coverage window
    result = json.loads(await count_launches_in_year_logic(client, 2022))
    assert result["data_available"] is True
    assert result["year"] == 2022
    call = client.query_launches_calls[-1]
    assert call["query"]["date_utc"]["$gte"] == "2022-01-01T00:00:00.000Z"
    assert call["query"]["date_utc"]["$lt"] == "2023-01-01T00:00:00.000Z"
    assert "docs" in result
    assert "totalDocs" in result


@pytest.mark.asyncio
async def test_count_launches_in_year_logic_out_of_range_returns_no_data() -> None:
    client = MockSpaceXClient()
    # 2024 is beyond the API data cutoff — should short-circuit without an API call
    result = json.loads(await count_launches_in_year_logic(client, 2024))
    assert result["data_available"] is False
    assert result["year"] == 2024
    assert result["totalDocs"] == 0
    assert "reason" in result
    # No API query should have been made
    assert len(client.query_launches_calls) == 0


@pytest.mark.asyncio
async def test_search_launches_logic_queries_by_name() -> None:
    client = MockSpaceXClient()
    result = json.loads(await search_launches_logic(client, "Starlink"))
    call = client.query_launches_calls[-1]
    assert call["query"]["name"]["$regex"] == "Starlink"
    assert call["query"]["name"]["$options"] == "i"
    assert "docs" in result


@pytest.mark.asyncio
async def test_get_launches_by_rocket_logic_resolves_rocket_id() -> None:
    client = MockSpaceXClient()
    result = json.loads(await get_launches_by_rocket_logic(client, "Falcon 9"))
    call = client.query_launches_calls[-1]
    assert MockSpaceXClient._FALCON_9_ID in call["query"]["rocket"]["$in"]
    assert "docs" in result


@pytest.mark.asyncio
async def test_get_launches_by_rocket_logic_with_successful_only() -> None:
    client = MockSpaceXClient()
    await get_launches_by_rocket_logic(client, "Falcon 9", successful_only=True)
    call = client.query_launches_calls[-1]
    assert call["query"]["success"] is True
    assert MockSpaceXClient._FALCON_9_ID in call["query"]["rocket"]["$in"]


@pytest.mark.asyncio
async def test_get_launches_by_rocket_logic_no_match_returns_error() -> None:
    client = MockSpaceXClient()
    result = json.loads(await get_launches_by_rocket_logic(client, "Starship"))
    assert "error" in result
    assert "rockets_available" in result


@pytest.mark.asyncio
async def test_get_launches_from_location_logic_resolves_pad_id() -> None:
    client = MockSpaceXClient()
    result = json.loads(await get_launches_from_location_logic(client, "Vandenberg"))
    call = client.query_launches_calls[-1]
    assert MockSpaceXClient._VANDENBERG_ID in call["query"]["launchpad"]["$in"]
    assert "docs" in result


@pytest.mark.asyncio
async def test_get_launches_from_location_logic_no_match_returns_error() -> None:
    client = MockSpaceXClient()
    result = json.loads(await get_launches_from_location_logic(client, "Baikonur"))
    assert "error" in result
    assert "launchpads_available" in result
