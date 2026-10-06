"""Pluggable backends: models, protocols, registry, and the default embedder."""

from frictiontune.backends.embedding import OpenAiEmbedder
from frictiontune.backends.models import (
    BackendModel,
    ExportRequest,
    ExportResult,
    GrpoRequest,
    ParityResult,
    ProbeExample,
    RolloutRequest,
    RolloutResult,
    RolloutStep,
    TrainingRequest,
    TrainingResult,
)
from frictiontune.backends.protocols import (
    Embedder,
    RolloutBackend,
    ServingBackend,
    TrainingBackend,
)
from frictiontune.backends.registry import BackendRegistry

__all__ = [
    "BackendModel",
    "BackendRegistry",
    "Embedder",
    "ExportRequest",
    "ExportResult",
    "GrpoRequest",
    "OpenAiEmbedder",
    "ParityResult",
    "ProbeExample",
    "RolloutBackend",
    "RolloutRequest",
    "RolloutResult",
    "RolloutStep",
    "ServingBackend",
    "TrainingBackend",
    "TrainingRequest",
    "TrainingResult",
]
