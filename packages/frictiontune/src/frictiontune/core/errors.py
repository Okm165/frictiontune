"""Typed exception hierarchy for frictiontune.

All errors raised by the library derive from :class:`FrictionTuneError`, so the
CLI and the orchestrator can catch a single root type while still branch on the
specific failure class.
"""

from __future__ import annotations

from typing import Any


class FrictionTuneError(Exception):
    """Root of every error raised by frictiontune."""


class ConfigError(FrictionTuneError):
    """Configuration is missing, malformed, or internally inconsistent."""


class StageError(FrictionTuneError):
    """A pipeline stage failed to execute."""

    def __init__(self, stage: str, message: str) -> None:
        """Record the failing stage and message."""
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class GateError(FrictionTuneError):
    """A stage gate (tier or kill criterion) prevents the run from continuing."""


class ValidationError(FrictionTuneError):
    """A typed boundary received data that could not be validated or repaired."""

    def __init__(self, model: str, attempts: int, detail: str) -> None:
        """Record the model, attempt count, and failure detail."""
        self.model = model
        self.attempts = attempts
        super().__init__(f"validation failed for {model} after {attempts} attempt(s): {detail}")


class PromptError(FrictionTuneError):
    """A prompt could not be loaded, parsed, or rendered."""


class LlmError(FrictionTuneError):
    """An LLM call failed after exhausting the configured retry policy."""


class LlmSchemaError(FrictionTuneError):
    """The LLM returned output that could not be parsed into the requested schema."""


class ArtifactError(FrictionTuneError):
    """An artifact could not be read, written, or reconciled."""


class IngestError(FrictionTuneError):
    """The source corpus could not be read or parsed."""


class BackendUnavailableError(FrictionTuneError):
    """An optional backend (e.g. a training stack) is not installed."""

    def __init__(self, backend: str, hint: str) -> None:
        """Record the backend name and its install hint."""
        self.backend = backend
        self.hint = hint
        super().__init__(f"backend {backend!r} is unavailable: {hint}")


class UnsupportedOperationError(FrictionTuneError):
    """A code path exists only for documentation and must not be executed."""

    def __init__(
        self, operation: str, *, reason: str, payload: dict[str, Any] | None = None
    ) -> None:
        """Record the forbidden operation, its reason, and any payload."""
        self.operation = operation
        self.reason = reason
        self.payload = payload or {}
        super().__init__(f"unsupported operation {operation!r}: {reason}")
