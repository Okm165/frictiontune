"""Structural contracts for pluggable backends.

Core defines the protocols; the ``frictiontune-train`` package provides the
concrete training/serving/rollout implementations. Keeping the protocols here
means the data pipeline never imports a heavy ML stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.backends.models import (
        DpoRequest,
        ExportRequest,
        ExportResult,
        GrpoRequest,
        RolloutRequest,
        RolloutResult,
        TrainingRequest,
        TrainingResult,
    )


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors (retrieval, calibration, clustering)."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving order."""
        ...


@runtime_checkable
class TrainingBackend(Protocol):
    """Runs supervised (and later RL) fine-tuning."""

    async def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute one supervised training job and return its artifacts."""
        ...

    async def train_grpo(self, request: GrpoRequest) -> TrainingResult:
        """Execute one GRPO job and return its artifacts."""
        ...

    async def train_dpo(self, request: DpoRequest) -> TrainingResult:
        """Execute one DPO preference job and return its artifacts."""
        ...


@runtime_checkable
class ServingBackend(Protocol):
    """Merges or packages an adapter for serving."""

    async def export(self, request: ExportRequest) -> ExportResult:
        """Produce a servable artifact and its parity report."""
        ...


@runtime_checkable
class RolloutBackend(Protocol):
    """Runs an agent against a sandbox for simulation/rollout."""

    async def rollout(self, request: RolloutRequest) -> RolloutResult:
        """Execute one rollout and return the transcript."""
        ...
