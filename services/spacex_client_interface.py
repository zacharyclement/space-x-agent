"""Abstract SpaceX API client contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal

LaunchRecord = Mapping[str, object]
QueryResponse = Mapping[str, object]


class SpaceXClientInterface(ABC):
    """Interface for querying SpaceX launch and rocket data."""

    @abstractmethod
    async def get_latest_launch(self) -> LaunchRecord:
        """Fetch the latest launch."""

    @abstractmethod
    async def get_next_launch(self) -> LaunchRecord:
        """Fetch the next scheduled launch."""

    @abstractmethod
    async def get_launches(
        self,
        *,
        year: int | None = None,
        successful: bool | None = None,
        limit: int = 100,
    ) -> Sequence[LaunchRecord]:
        """Fetch launches with optional filters."""

    @abstractmethod
    async def search_launches(self, query: str, *, limit: int = 10) -> Sequence[LaunchRecord]:
        """Search launches by mission name."""

    @abstractmethod
    async def get_rocket(self, rocket_id: str) -> LaunchRecord:
        """Fetch a rocket by ID."""

    @abstractmethod
    async def get_successful_launches_by_rocket(
        self, rocket_name: str, *, limit: int = 10
    ) -> Sequence[LaunchRecord]:
        """Fetch successful launches for a given rocket name."""

    @abstractmethod
    async def query_launches_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 10,
        populate_rocket: bool = True,
        populate_launchpad: bool = False,
        sort_direction: Literal["asc", "desc"] = "desc",
    ) -> QueryResponse:
        """Return raw `/launches/query` response payload."""

    @abstractmethod
    async def query_rockets_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 10,
    ) -> QueryResponse:
        """Return raw `/rockets/query` response payload."""

    @abstractmethod
    async def close(self) -> None:
        """Release HTTP resources."""
