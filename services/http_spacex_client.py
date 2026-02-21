"""HTTP implementation of the SpaceX client interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import httpx

from core.exceptions import SpaceXApiError
from services.spacex_client_interface import LaunchRecord, SpaceXClientInterface


class HttpSpaceXClient(SpaceXClientInterface):
    """SpaceX client backed by the public REST API."""

    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        """Initialize the async HTTP client.

        Args:
            base_url: Base URL for the SpaceX API.
            timeout_seconds: Request timeout.
        """
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )

    async def get_latest_launch(self) -> LaunchRecord:
        """Fetch the latest SpaceX launch."""
        response = await self._request_json("GET", "/launches/latest")
        return self._as_mapping(response, message="Unexpected payload for latest launch.")

    async def get_next_launch(self) -> LaunchRecord:
        """Fetch the next planned SpaceX launch."""
        response = await self._request_json("GET", "/launches/next")
        return self._as_mapping(response, message="Unexpected payload for next launch.")

    async def get_launches(
        self,
        *,
        year: int | None = None,
        successful: bool | None = None,
        limit: int = 100,
    ) -> Sequence[LaunchRecord]:
        """Fetch launches with optional year and success filters."""
        query: dict[str, object] = {}
        if year is not None:
            query["date_utc"] = {
                "$gte": f"{year}-01-01T00:00:00.000Z",
                "$lt": f"{year + 1}-01-01T00:00:00.000Z",
            }
        if successful is not None:
            query["success"] = successful

        return await self._query_launches(query=query, limit=limit, populate_rocket=True)

    async def search_launches(self, query: str, *, limit: int = 10) -> Sequence[LaunchRecord]:
        """Search launches by mission name."""
        query_filter: dict[str, object] = {
            "name": {"$regex": query, "$options": "i"},
        }
        return await self._query_launches(query=query_filter, limit=limit, populate_rocket=True)

    async def get_rocket(self, rocket_id: str) -> LaunchRecord:
        """Fetch a rocket by ID."""
        response = await self._request_json("GET", f"/rockets/{rocket_id}")
        return self._as_mapping(response, message="Unexpected payload for rocket.")

    async def get_successful_launches_by_rocket(
        self, rocket_name: str, *, limit: int = 10
    ) -> Sequence[LaunchRecord]:
        """Fetch successful launches for a rocket name."""
        rocket_query = {
            "query": {"name": {"$regex": rocket_name, "$options": "i"}},
            "options": {"limit": 1},
        }
        rocket_response = await self._request_json(
            "POST",
            "/rockets/query",
            json_payload=rocket_query,
        )
        rocket_docs = self._extract_docs(
            rocket_response, message="Unexpected payload while searching rocket."
        )
        if not rocket_docs:
            return []
        rocket_id = rocket_docs[0].get("id")
        if not isinstance(rocket_id, str):
            return []
        return await self._query_launches(
            query={"rocket": rocket_id, "success": True},
            limit=limit,
            populate_rocket=True,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _query_launches(
        self,
        *,
        query: Mapping[str, object],
        limit: int,
        populate_rocket: bool,
    ) -> Sequence[LaunchRecord]:
        options: dict[str, object] = {"limit": limit, "sort": {"date_utc": "desc"}}
        if populate_rocket:
            options["populate"] = [{"path": "rocket", "select": "name type"}]
        payload = {"query": dict(query), "options": options}
        response = await self._request_json("POST", "/launches/query", json_payload=payload)
        return self._extract_docs(response, message="Unexpected payload for launch query.")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, object] | None = None,
    ) -> object:
        try:
            response = await self._client.request(method, path, json=json_payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise SpaceXApiError(
                f"SpaceX API returned HTTP {exc.response.status_code} for {method} {path}.",
                status_code=exc.response.status_code,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise SpaceXApiError(
                f"SpaceX API request failed for {method} {path}.",
                cause=exc,
            ) from exc
        except ValueError as exc:
            raise SpaceXApiError("SpaceX API returned non-JSON response.", cause=exc) from exc

    @staticmethod
    def _as_mapping(payload: object, *, message: str) -> Mapping[str, object]:
        if not isinstance(payload, Mapping):
            raise SpaceXApiError(message)
        return payload

    @staticmethod
    def _extract_docs(payload: object, *, message: str) -> Sequence[LaunchRecord]:
        if not isinstance(payload, Mapping):
            raise SpaceXApiError(message)
        docs = payload.get("docs")
        if not isinstance(docs, list):
            raise SpaceXApiError(message)
        return [doc for doc in docs if isinstance(doc, Mapping)]
