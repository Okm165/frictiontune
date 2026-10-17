"""Run manifests: the machine-readable record of what a run produced."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.artifacts import DatasetRef
from frictiontune.pipeline.spec import StageSpec
from frictiontune.version import __version__

# Only source can change a stage's behaviour in a way the *configuration*
# fingerprint does not already capture. Configuration identity is derived from
# the resolved config itself (`RunConfig.semantic_fingerprint`, scoped to the
# stage's own settings block and the roles it declares), so hashing `configs/`
# here as well made every edit re-key *every* stage — changing `export`'s parity
# probes invalidated `ingest`. A doc or an unrelated stage's knob must not
# invalidate artifacts, and a knob that matters is already in the run id.
_DIRTY_SCOPE = ("packages",)
# `configs/` also holds prose (READMEs, design notes). Exclude it so editing a
# doc under `configs/` honours the rule above instead of invalidating every
# stage's artifact — a doc change cannot alter a stage's behaviour.
_DOC_EXCLUDES = (":(exclude)**/*.md",)
_DIRTY_PATHS = (*_DIRTY_SCOPE, *_DOC_EXCLUDES)


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _git(*args: str) -> str | None:
    """Return trimmed stdout of a git command, or ``None`` if it cannot run."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _dirty_fingerprint() -> str:
    """Digest the *content* of every uncommitted source/config change.

    Listing the changed paths is not enough: editing a file that is already
    dirty leaves the path list unchanged, so a stage whose code changed would
    keep the same run id and silently reuse an artifact from the previous code.
    Hashing the diff (and the bytes of new untracked files) makes any edit — or
    addition, or deletion — change the run id.
    """
    hasher = hashlib.sha256()
    diff = _git("diff", "HEAD", "--", *_DIRTY_PATHS)
    if diff:
        hasher.update(diff.encode("utf-8"))
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *_DIRTY_PATHS)
    for relative in sorted(line for line in (untracked or "").splitlines() if line.strip()):
        hasher.update(relative.encode("utf-8"))
        try:
            hasher.update(Path(relative).read_bytes())
        except OSError:
            hasher.update(b"<unreadable>")
    return hasher.hexdigest()[:12]


def code_revision() -> str:
    """Best-effort code revision, including uncommitted working-tree changes.

    The commit alone is not enough: during development the tree is dirty, and a
    stage whose *code* changed would otherwise keep the same run id and silently
    reuse an artifact produced by the previous code (this once let a removed
    model field shadow a fresh artifact until schema validation failed). So when
    source or config files differ from ``HEAD``, a digest of their contents is
    appended, which changes the run id and forces the affected stages to re-run.
    """
    head = _git("rev-parse", "HEAD")
    if head is None or not head.strip():
        return f"frictiontune/{__version__}"
    status = _git("status", "--porcelain", "--", *_DIRTY_PATHS)
    if status is None or not status.strip():
        return head.strip()
    return f"{head.strip()}+dirty.{_dirty_fingerprint()}"


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
