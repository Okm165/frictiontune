"""Structured logging.

One place configures the process-wide logger. Every module asks for a bound
logger via :func:`get_logger`. Human-readable console output is preferred; JSON
is available for machine tails. Debugging is expected, so ``debug`` fields are
first-class and never stripped.
"""

from __future__ import annotations

import logging as _stdlib_logging
import sys
from typing import Any, Literal, cast

import structlog

LogFormat = Literal["console", "json"]
LogLevel = Literal["debug", "info", "warning", "error", "critical"]

_LEVELS: dict[LogLevel, int] = {
    "debug": _stdlib_logging.DEBUG,
    "info": _stdlib_logging.INFO,
    "warning": _stdlib_logging.WARNING,
    "error": _stdlib_logging.ERROR,
    "critical": _stdlib_logging.CRITICAL,
}

_CONFIGURED = False


def configure_logging(
    *,
    level: LogLevel = "info",
    log_format: LogFormat = "console",
) -> None:
    """Configure structlog once for the whole process.

    Safe to call repeatedly; subsequent calls are no-ops so a library consumer
    can own the configuration instead.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    renderer: Any
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LEVELS[level]),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to the module ``name``."""
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
