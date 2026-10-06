"""Lazy imports with actionable errors.

Every heavy dependency is imported inside a function through :func:`require`, so
``import frictiontune_train`` is safe on a machine without torch and a missing
dependency produces a one-line fix rather than a traceback.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypeVar

from frictiontune.core.errors import BackendUnavailableError

T = TypeVar("T")


def require(module: str, *, hint: str) -> Any:
    """Import ``module`` or raise a :class:`BackendUnavailableError` with a hint."""
    try:
        return import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BackendUnavailableError(module, hint) from exc


def require_attr(module: str, name: str, *, hint: str) -> Any:
    """Import ``module`` and return attribute ``name``."""
    loaded = require(module, hint=hint)
    try:
        return getattr(loaded, name)
    except AttributeError as exc:  # pragma: no cover - version drift
        raise BackendUnavailableError(
            module, f"{module} has no attribute {name!r}; {hint}"
        ) from exc


def torch_hint() -> str:
    """The standard install hint for the training extras."""
    return "uv sync --package frictiontune-train --extra train"
