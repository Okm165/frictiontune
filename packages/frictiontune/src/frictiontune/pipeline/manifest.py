"""Run manifests: the machine-readable record of what a run produced."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.artifacts import DatasetRef
from frictiontune.pipeline.spec import StageSpec
from frictiontune.version import __version__

if TYPE_CHECKING:
    from pathlib import Path


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def code_revision() -> str:
    """Best-effort code revision: the git commit if available, else the version."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return f"frictiontune/{__version__}"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return f"frictiontune/{__version__}"


class StageManifest(BaseModel):
    """The record of one stage invocation."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    run_id: str
    spec: StageSpec
    code_revision: str
    config_fingerprint: str
    inputs: dict[str, DatasetRef] = Field(default_factory=dict)
    outputs: dict[str, DatasetRef] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str
    duration_s: float
    cached: bool = False


class RunManifest(BaseModel):
    """The record of a whole pipeline run."""

    model_config = ConfigDict(extra="forbid")

    pipeline: str
    project: str
    code_revision: str
    config_fingerprint: str
    artifacts_root: str
    started_at: str
    finished_at: str
    stages: list[StageManifest] = Field(default_factory=list)

    def stage_names(self) -> list[str]:
        """The names of the stages that ran."""
        return [entry.stage for entry in self.stages]


def write_stage_manifest(path: Path, manifest: StageManifest) -> None:
    """Write a stage manifest as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_stage_manifest(path: Path) -> StageManifest:
    """Read a stage manifest back into its typed form."""
    return StageManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Write the run manifest as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_run_manifest(path: Path) -> RunManifest:
    """Read a run manifest back into its typed form."""
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
