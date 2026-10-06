"""Registry for pluggable backends.

Backends are looked up by capability. A stage asks for a training backend and,
if none is installed, gets a :class:`BackendUnavailableError` with an actionable
hint rather than an obscure import error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.core.errors import BackendUnavailableError

if TYPE_CHECKING:
    from frictiontune.backends.protocols import (
        Embedder,
        RolloutBackend,
        ServingBackend,
        TrainingBackend,
    )


class BackendRegistry:
    """Holds the active backend implementations for a run."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._embedder: Embedder | None = None
        self._training: TrainingBackend | None = None
        self._serving: ServingBackend | None = None
        self._rollout: RolloutBackend | None = None

    # -- registration ------------------------------------------------------

    def set_embedder(self, embedder: Embedder) -> None:
        """Register the embedding backend."""
        self._embedder = embedder

    def set_training(self, backend: TrainingBackend) -> None:
        """Register the training backend."""
        self._training = backend

    def set_serving(self, backend: ServingBackend) -> None:
        """Register the serving/export backend."""
        self._serving = backend

    def set_rollout(self, backend: RolloutBackend) -> None:
        """Register the rollout backend."""
        self._rollout = backend

    # -- access ------------------------------------------------------------

    def embedder(self) -> Embedder:
        """Return the embedding backend or fail with a hint."""
        if self._embedder is None:
            raise BackendUnavailableError(
                "embedder", "configure an embedding provider in roles/config"
            )
        return self._embedder

    def training(self) -> TrainingBackend:
        """Return the training backend or fail with a hint."""
        if self._training is None:
            raise BackendUnavailableError(
                "training",
                "install frictiontune-train (uv sync --package frictiontune-train --extra train)",
            )
        return self._training

    def serving(self) -> ServingBackend:
        """Return the serving backend or fail with a hint."""
        if self._serving is None:
            raise BackendUnavailableError(
                "serving",
                "install frictiontune-train (uv sync --package frictiontune-train --extra train)",
            )
        return self._serving

    def rollout(self) -> RolloutBackend:
        """Return the rollout backend or fail with a hint."""
        if self._rollout is None:
            raise BackendUnavailableError(
                "rollout",
                "install frictiontune-train and enable a rollout sandbox",
            )
        return self._rollout

    def embedder_or_none(self) -> Embedder | None:
        """Return the embedding backend if one is registered, else ``None``."""
        return self._embedder
