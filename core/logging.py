"""Structured logging setup."""

from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str) -> None:
    """Configure stdlib and structlog for JSON-friendly structured logging.

    Args:
        log_level: Logging level such as INFO or DEBUG.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a configured structured logger.

    Args:
        name: Logger namespace.

    Returns:
        Bound structured logger.
    """
    return structlog.get_logger(name)
