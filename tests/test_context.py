"""Tests for the per-stage artifact layout.

A pipeline's stages must nest under the pipeline directory so the CLI
(``inspect``/``dashboard``) can find them: ``artifacts/<pipeline>/<stage>/<run_id>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from frictiontune.pipeline.context import RunContext

if TYPE_CHECKING:
    from pathlib import Path


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        config=cast("Any", object()),
        pipeline="core",
        cache=cast("Any", object()),
        gateway=cast("Any", object()),
        backends=cast("Any", object()),
        logger=cast("Any", object()),
        artifacts_root=tmp_path,
    )


def test_stage_store_nests_under_the_pipeline(tmp_path: Path) -> None:
    store = _context(tmp_path).store("ingest", "abc123")

    assert store.directory == tmp_path / "core" / "ingest" / "abc123"
    assert (store.directory / "data").is_dir()
