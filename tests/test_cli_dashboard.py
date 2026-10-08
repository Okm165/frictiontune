"""Tests for selecting which evaluate_replay dashboard the CLI renders.

Run ids are content hashes with no temporal order, so the command must not rely
on a lexical scan; it should prefer the run the manifest names and fall back to
the most recently modified directory.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from frictiontune.cli import _latest_dashboard
from frictiontune.pipeline.manifest import RunManifest, StageManifest, write_run_manifest
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier

if TYPE_CHECKING:
    from pathlib import Path

_SPEC = StageSpec(
    name="evaluate_replay",
    tier=Tier.A,
    lead=Lead.CORE,
    roi="High",
    justification="j",
    evidence=[],
    cost=CostProfile(),
)


def _dashboard(root: Path, pipeline: str, run_id: str) -> Path:
    run_dir = root / pipeline / "evaluate_replay" / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "dashboard.json"
    path.write_text(json.dumps({"signals": []}), encoding="utf-8")
    return path


def _manifest(root: Path, pipeline: str, run_id: str) -> None:
    entry = StageManifest(
        stage="evaluate_replay",
        run_id=run_id,
        spec=_SPEC,
        code_revision="r",
        config_fingerprint="c",
        started_at="t",
        finished_at="t",
        duration_s=0.0,
    )
    manifest = RunManifest(
        pipeline=pipeline,
        project="p",
        code_revision="r",
        config_fingerprint="c",
        artifacts_root=str(root),
        started_at="t",
        finished_at="t",
        stages=[entry],
    )
    write_run_manifest(root / pipeline / "manifest.json", manifest)


def test_dashboard_prefers_the_run_named_by_the_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    older = _dashboard(root, "dev", "aaaarun")
    newer = _dashboard(root, "dev", "zzzzrun")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    _manifest(root, "dev", "aaaarun")

    assert _latest_dashboard(root / "dev") == older


def test_dashboard_falls_back_to_the_newest_directory(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    older = _dashboard(root, "dev", "aaaarun")
    newer = _dashboard(root, "dev", "zzzzrun")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    assert _latest_dashboard(root / "dev") == newer


def test_dashboard_returns_none_when_absent(tmp_path: Path) -> None:
    assert _latest_dashboard(tmp_path / "artifacts" / "dev") is None
