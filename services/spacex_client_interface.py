"""Abstract SpaceX API client contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SpaceXClientInterface(ABC):
    """Interface for querying SpaceX data from the public REST API."""

    @abstractmethod
    async def get_latest_launch(self) -> dict:
        """Fetch the most recent past launch.

        Returns:
            Raw JSON response from GET /launches/latest.
        """

    @abstractmethod
    async def get_next_launch(self) -> dict:
        """Fetch the next scheduled upcoming launch.

        Returns:
            Raw JSON response from GET /launches/next.
        """

    @abstractmethod
    async def query_launches(
        self,
        query: dict,
        *,
        options: dict | None = None,
    ) -> dict:
        """Query launches via the SpaceX /launches/query endpoint.

        Args:
            query: MongoDB-style filter document.
            options: Pagination/sort/populate options (limit, sort, populate, etc.).

        Returns:
            Raw JSON response from POST /launches/query containing ``docs`` and ``totalDocs``.
        """

    @abstractmethod
    async def get_rockets(self) -> list[dict]:
        """Fetch all rockets.

        Returns:
            Raw JSON list from GET /rockets.
        """

    @abstractmethod
    async def get_launchpads(self) -> list[dict]:
        """Fetch all launchpads.

        Returns:
            Raw JSON list from GET /launchpads.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release HTTP resources."""
