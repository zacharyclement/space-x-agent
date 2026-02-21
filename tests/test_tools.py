"""Tests for SpaceX tool logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from services.spacex_client_interface import LaunchRecord, SpaceXClientInterface
from tools.spacex_tools import (
    count_successful_launches_tool_logic,
    latest_launch_tool_logic,
    mission_rocket_tool_logic,
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


@pytest.mark.asyncio
async def test_latest_launch_tool_logic_formats_output() -> None:
    client = MockSpaceXClient()
    result = await latest_launch_tool_logic(client)
    assert "Latest launch:" in result
    assert "Starlink 9-1" in result


@pytest.mark.asyncio
async def test_count_successful_launches_tool_logic() -> None:
    client = MockSpaceXClient()
    result = await count_successful_launches_tool_logic(client, 2024)
    assert "5" in result
    assert "2024" in result


@pytest.mark.asyncio
async def test_mission_rocket_tool_logic() -> None:
    client = MockSpaceXClient()
    result = await mission_rocket_tool_logic(client, "Starlink 9-1")
    assert "Falcon 9" in result


@pytest.mark.asyncio
async def test_search_launches_no_match() -> None:
    client = MockSpaceXClient()
    result = await search_launches_tool_logic(client, "missing")
    assert "No launches found" in result


@pytest.mark.asyncio
async def test_successful_launches_for_rocket() -> None:
    client = MockSpaceXClient()
    result = await successful_launches_for_rocket_tool_logic(client, "Falcon 9", limit=2)
    assert "Successful launches for 'Falcon 9'" in result
    assert "Falcon Mission 1" in result
