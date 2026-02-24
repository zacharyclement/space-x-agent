"""HTTP implementation of the SpaceX client interface."""

from __future__ import annotations

import httpx

from core.exceptions import SpaceXApiError
from core.logging import get_logger
from services.spacex_client_interface import SpaceXClientInterface

logger = get_logger(__name__)


class HttpSpaceXClient(SpaceXClientInterface):
    """SpaceX client backed by the public REST API."""

    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        """Initialize the async HTTP client.

        Args:
            base_url: Base URL for the SpaceX API (e.g. ``https://api.spacexdata.com/v5``).
            timeout_seconds: Request timeout in seconds.
        """
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )
        self._logger = logger.bind(component="spacex_http_client", base_url=base_url)

    async def get_latest_launch(self) -> dict:
        """Fetch the most recent past launch from GET /launches/latest."""
        self._logger.debug("spacex_get_latest_launch_started")
        data = await self._get("/launches/latest")
        name = data.get("name") if isinstance(data, dict) else None
        self._logger.debug("spacex_get_latest_launch_succeeded", name=name)
        return self._as_dict(data, endpoint="/launches/latest")

    async def get_next_launch(self) -> dict:
        """Fetch the next scheduled launch from GET /launches/next."""
        self._logger.debug("spacex_get_next_launch_started")
        data = await self._get("/launches/next")
        name = data.get("name") if isinstance(data, dict) else None
        self._logger.debug("spacex_get_next_launch_succeeded", name=name)
        return self._as_dict(data, endpoint="/launches/next")

    async def query_launches(
        self,
        query: dict,
        *,
        options: dict | None = None,
    ) -> dict:
        """POST /launches/query with the provided filter and options.

        Args:
            query: MongoDB-style filter document.
            options: Pagination/sort/populate options.

        Returns:
            Raw ``{"docs": [...], "totalDocs": N, ...}`` response.
        """
        body: dict = {"query": query, "options": options or {}}
        self._logger.debug("spacex_query_launches_started", query_keys=sorted(query.keys()))
        data = await self._post("/launches/query", body)
        result = self._as_dict(data, endpoint="/launches/query")
        docs = result.get("docs")
        self._logger.debug(
            "spacex_query_launches_succeeded",
            total_docs=result.get("totalDocs"),
            returned_docs=len(docs) if isinstance(docs, list) else 0,
        )
        return result

    async def get_rockets(self) -> list[dict]:
        """Fetch all rockets from GET /rockets."""
        self._logger.debug("spacex_get_rockets_started")
        data = await self._get("/rockets")
        if not isinstance(data, list):
            raise SpaceXApiError("Expected a list from GET /rockets.")
        self._logger.debug("spacex_get_rockets_succeeded", count=len(data))
        return [item for item in data if isinstance(item, dict)]

    async def get_launchpads(self) -> list[dict]:
        """Fetch all launchpads from GET /launchpads."""
        self._logger.debug("spacex_get_launchpads_started")
        data = await self._get("/launchpads")
        if not isinstance(data, list):
            raise SpaceXApiError("Expected a list from GET /launchpads.")
        self._logger.debug("spacex_get_launchpads_succeeded", count=len(data))
        return [item for item in data if isinstance(item, dict)]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> object:
        """Send a GET request and return parsed JSON."""
        return await self._request("GET", path)

    async def _post(self, path: str, body: dict) -> object:
        """Send a POST request with a JSON body and return parsed JSON."""
        return await self._request("POST", path, json_body=body)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
    ) -> object:
        self._logger.debug("spacex_http_request_started", method=method, path=path)
        response: httpx.Response | None = None
        try:
            response = await self._client.request(method, path, json=json_body)
            response.raise_for_status()
            data = response.json()
            self._logger.debug(
                "spacex_http_request_succeeded",
                method=method,
                path=path,
                status_code=response.status_code,
            )
            return data
        except httpx.HTTPStatusError as exc:
            self._logger.error(
                "spacex_http_status_error",
                method=method,
                path=path,
                status_code=exc.response.status_code,
                response_text=exc.response.text[:500],
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
            status_code = response.status_code if response is not None else None
            self._logger.error(
                "spacex_http_non_json_response",
                method=method,
                path=path,
                status_code=status_code,
            )
            raise SpaceXApiError("SpaceX API returned a non-JSON response.", cause=exc) from exc

    @staticmethod
    def _as_dict(payload: object, *, endpoint: str) -> dict:
        if not isinstance(payload, dict):
            raise SpaceXApiError(
                f"Expected a JSON object from {endpoint}, got {type(payload).__name__}."
            )
        return payload
