"""Domain-specific exception types for the SpaceX agent."""

from __future__ import annotations


class SpaceXAgentError(Exception):
    """Base exception for all agent-specific failures."""


class ConfigurationError(SpaceXAgentError):
    """Raised when application configuration is invalid."""


class SpaceXApiError(SpaceXAgentError):
    """Raised when the SpaceX API cannot be queried successfully."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a SpaceX API error.

        Args:
            message: Human-readable error message.
            status_code: Optional HTTP status code from the upstream API.
            cause: Optional underlying exception.
        """
        super().__init__(message)
        self.status_code = status_code
        self.cause = cause
