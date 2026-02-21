"""HTTP implementation of the SpaceX client interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import httpx

from core.exceptions import SpaceXApiError
from core.logging import get_logger
from services.spacex_client_interface import LaunchRecord, QueryResponse, SpaceXClientInterface

logger = get_logger(__name__)


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
        self._logger = logger.bind(component="spacex_http_client", base_url=base_url)

    async def get_latest_launch(self) -> LaunchRecord:
        """Fetch the latest SpaceX launch with populated related records when possible."""
        launches = await self._query_launches(
            query={"upcoming": False},
            limit=1,
            populate_rocket=True,
            populate_launchpad=True,
            sort_direction="desc",
        )
        if launches:
            self._logger.debug(
                "spacex_latest_launch_selected",
                source="launches/query",
                launch=self._launch_snapshot(launches[0]),
            )
            return launches[0]

        self._logger.warning(
            "spacex_latest_launch_query_empty_fallback",
            fallback_endpoint="/launches/latest",
        )
        response = await self._request_json("GET", "/launches/latest")
        launch = self._as_mapping(response, message="Unexpected payload for latest launch.")
        self._logger.debug(
            "spacex_latest_launch_selected",
            source="launches/latest",
            launch=self._launch_snapshot(launch),
        )
        return launch

    async def get_next_launch(self) -> LaunchRecord:
        """Fetch the next planned SpaceX launch with launchpad details when possible."""
        launches = await self._query_launches(
            query={"upcoming": True},
            limit=1,
            populate_rocket=True,
            populate_launchpad=True,
            sort_direction="asc",
        )
        if launches:
            self._logger.debug(
                "spacex_next_launch_selected",
                source="launches/query",
                launch=self._launch_snapshot(launches[0]),
            )
            return launches[0]

        self._logger.warning(
            "spacex_next_launch_query_empty_fallback",
            fallback_endpoint="/launches/next",
        )
        response = await self._request_json("GET", "/launches/next")
        launch = self._as_mapping(response, message="Unexpected payload for next launch.")
        self._logger.debug(
            "spacex_next_launch_selected",
            source="launches/next",
            launch=self._launch_snapshot(launch),
        )
        return launch

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
        rocket_response = await self.query_rockets_raw(
            query={"name": {"$regex": rocket_name, "$options": "i"}},
            limit=1,
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
        """Return raw `/launches/query` response payload."""
        options: dict[str, object] = {"limit": limit, "sort": {"date_utc": sort_direction}}
        if select_fields:
            options["select"] = select_fields
        if populate_rocket:
            options["populate"] = [{"path": "rocket", "select": "name type"}]
        if populate_launchpad:
            populate = options.setdefault("populate", [])
            if isinstance(populate, list):
                populate.append({"path": "launchpad", "select": "name full_name locality region"})

        payload = {"query": dict(query), "options": options}
        self._logger.debug("spacex_launch_query_started", payload=self._summarize_payload(payload))
        response = await self._request_json("POST", "/launches/query", json_payload=payload)
        mapping = self._as_mapping(response, message="Unexpected payload for launch query.")
        docs = self._extract_docs(mapping, message="Unexpected payload for launch query.")
        self._logger.debug(
            "spacex_launch_query_succeeded",
            query_keys=sorted(query.keys()),
            returned_docs=len(docs),
            sample_launches=[self._launch_snapshot(launch) for launch in docs[:3]],
        )
        return mapping

    async def query_rockets_raw(
        self,
        *,
        query: Mapping[str, object],
        limit: int = 1000,
    ) -> QueryResponse:
        """Return broad raw rocket payload from `/rockets`."""
        del query, limit
        self._logger.debug("spacex_rocket_query_started", endpoint="/rockets")
        response = await self._request_json("GET", "/rockets")
        if isinstance(response, list):
            docs = [doc for doc in response if isinstance(doc, Mapping)]
            mapping: QueryResponse = {"docs": docs, "totalDocs": len(docs)}
            self._logger.debug(
                "spacex_rocket_query_succeeded",
                endpoint="/rockets",
                returned_docs=len(docs),
            )
            return mapping

        mapping = self._as_mapping(response, message="Unexpected payload for rocket query.")
        docs = self._extract_docs(mapping, message="Unexpected payload for rocket query.")
        self._logger.debug(
            "spacex_rocket_query_succeeded",
            endpoint="/rockets",
            returned_docs=len(docs),
        )
        return mapping

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _query_launches(
        self,
        *,
        query: Mapping[str, object],
        limit: int,
        populate_rocket: bool,
        populate_launchpad: bool = False,
        sort_direction: Literal["asc", "desc"] = "desc",
    ) -> Sequence[LaunchRecord]:
        response = await self.query_launches_raw(
            query=query,
            limit=limit,
            populate_rocket=populate_rocket,
            populate_launchpad=populate_launchpad,
            sort_direction=sort_direction,
            select_fields=None,
        )
        return self._extract_docs(response, message="Unexpected payload for launch query.")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, object] | None = None,
    ) -> object:
        self._logger.debug(
            "spacex_http_request_started",
            method=method,
            path=path,
            payload=self._summarize_payload(json_payload),
        )
        response: httpx.Response | None = None
        try:
            response = await self._client.request(method, path, json=json_payload)
            response.raise_for_status()
            data = response.json()
            self._logger.debug(
                "spacex_http_request_succeeded",
                method=method,
                path=path,
                status_code=response.status_code,
                response=self._summarize_response(data),
            )
            return data
        except httpx.HTTPStatusError as exc:
            self._logger.error(
                "spacex_http_status_error",
                method=method,
                path=path,
                status_code=exc.response.status_code,
                response_text=self._trim_text(exc.response.text),
            )
            raise SpaceXApiError(
                f"SpaceX API returned HTTP {exc.response.status_code} for {method} {path}.",
                status_code=exc.response.status_code,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            self._logger.error(
                "spacex_http_transport_error",
                method=method,
                path=path,
                error=str(exc),
            )
            raise SpaceXApiError(
                f"SpaceX API request failed for {method} {path}.",
                cause=exc,
            ) from exc
        except ValueError as exc:
            self._logger.error(
                "spacex_http_non_json_response",
                method=method,
                path=path,
                status_code=response.status_code if response is not None else None,
                response_text=self._trim_text(response.text) if response is not None else "",
            )
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

    @staticmethod
    def _summarize_payload(payload: Mapping[str, object] | None) -> object:
        if payload is None:
            return None
        summary: dict[str, object] = {"keys": sorted(payload.keys())}
        query = payload.get("query")
        if isinstance(query, Mapping):
            summary["query_keys"] = sorted(query.keys())
        options = payload.get("options")
        if isinstance(options, Mapping):
            summary["options_keys"] = sorted(options.keys())
            limit = options.get("limit")
            if isinstance(limit, int):
                summary["limit"] = limit
            sort = options.get("sort")
            if isinstance(sort, Mapping):
                summary["sort"] = dict(sort)
        return summary

    @staticmethod
    def _summarize_response(payload: object) -> object:
        if isinstance(payload, Mapping):
            summary: dict[str, object] = {"keys": sorted(payload.keys())}
            docs = payload.get("docs")
            if isinstance(docs, list):
                summary["docs_count"] = len(docs)
            return summary
        if isinstance(payload, list):
            return {"type": "list", "count": len(payload)}
        return {"type": type(payload).__name__}

    @staticmethod
    def _launch_snapshot(launch: Mapping[str, object]) -> dict[str, object]:
        return {
            "name": launch.get("name"),
            "date_utc": launch.get("date_utc"),
            "success": launch.get("success"),
            "upcoming": launch.get("upcoming"),
            "launchpad": launch.get("launchpad"),
            "rocket": launch.get("rocket"),
        }

    @staticmethod
    def _trim_text(value: str, *, max_chars: int = 500) -> str:
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}..."
