"""Tests for the orchestrator's optional-input handling.

Optional inputs exist so a core stage (``judge``) can fold in data produced by a
gated extension without making the extension mandatory. These lock the two
properties that matter: an absent optional artifact is ignored, and a present
one changes the run id so a stale manifest is never reused.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from frictiontune.config.loader import load_config
from frictiontune.core.artifacts import DatasetRef
from frictiontune.core.hashing import file_sha256
from frictiontune.core.json import try_loads
from frictiontune.pipeline.context import ArtifactCatalog, RunContext
from frictiontune.pipeline.dag import DependencyGraph
from frictiontune.pipeline.orchestrator import Orchestrator
from frictiontune.pipeline.registry import StageRegistry
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult
from frictiontune.stages.judge import Judge

if TYPE_CHECKING:
    from frictiontune.pipeline.context import StageContext


class _Logger:
    """A do-nothing stand-in for the structlog logger."""

    def bind(self, **fields: object) -> _Logger:
        del fields
        return self

    def info(self, event: str, **fields: object) -> None:
        del event, fields


class _Gateway:
    """A stand-in gateway: only the prompt fingerprint is read by the orchestrator."""

    def prompt_fingerprint(self) -> str:
        return "test-prompts"


class _Probe(Stage):
    """A stage with one required and one optional input."""

    spec = StageSpec(
        name="probe",
        tier=Tier.A,
        lead=Lead.CORE,
        roi="High",
        justification="test double",
        evidence=["test"],
        cost=CostProfile(),
        inputs={"required_art": "jsonl"},
        optional_inputs={"optional_art": "jsonl"},
        outputs={"probe_out": "json"},
        kill_criterion="n/a",
    )

    async def run(self, ctx: StageContext) -> StageResult:
        """Report which inputs the catalog actually offered."""
        consumed = [
            *self.spec.inputs,
            *(key for key in self.spec.optional_inputs if ctx.catalog.has(key)),
        ]
        ctx.emit_json("probe_out", {"consumed": consumed})
        return StageResult.from_context(ctx, metrics={"consumed": float(len(consumed))})


def _publish(catalog: ArtifactCatalog, name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"x": 1}\n', encoding="utf-8")
    catalog.publish(
        DatasetRef(
            name=name,
            path=str(path),
            sha256=file_sha256(path),
            rows=1,
            kind="jsonl",
        ),
        kind="jsonl",
    )


def _execute(
    tmp_path: Path, config_dir: Path, *, with_optional: bool
) -> tuple[Any, dict[str, Any]]:
    catalog = ArtifactCatalog()
    _publish(catalog, "required_art", tmp_path / "required.jsonl")
    if with_optional:
        _publish(catalog, "optional_art", tmp_path / "optional.jsonl")

    registry = StageRegistry()
    registry.register(_Probe)
    run = RunContext(
        config=load_config(config_dir),
        pipeline="core",
        cache=cast("Any", object()),
        gateway=cast("Any", _Gateway()),
        backends=cast("Any", object()),
        logger=cast("Any", _Logger()),
        artifacts_root=tmp_path / "artifacts",
        catalog=catalog,
    )
    orchestrator = Orchestrator(run=run, registry=registry, graph=DependencyGraph({"probe": ()}))
    manifest = asyncio.run(orchestrator.execute(["probe"]))
    out_path = Path(manifest.stages[0].outputs["probe_out"].path)
    output = try_loads(out_path.read_text(encoding="utf-8"))
    return manifest, cast("dict[str, Any]", output)


def test_absent_optional_input_is_ignored(tmp_path: Path, config_dir: Path) -> None:
    manifest, output = _execute(tmp_path, config_dir, with_optional=False)

    assert output == {"consumed": ["required_art"]}
    assert set(manifest.stages[0].inputs) == {"required_art"}


def test_present_optional_input_is_consumed_and_changes_the_run_id(
    tmp_path: Path, config_dir: Path
) -> None:
    without, _ = _execute(tmp_path / "a", config_dir, with_optional=False)
    with_optional, output = _execute(tmp_path / "b", config_dir, with_optional=True)

    assert output == {"consumed": ["required_art", "optional_art"]}
    assert "optional_art" in with_optional.stages[0].inputs
    assert with_optional.stages[0].run_id != without.stages[0].run_id


def test_judge_declares_its_gated_data_as_optional() -> None:
    assert Judge.spec.inputs == {"candidates": "jsonl"}
    assert set(Judge.spec.optional_inputs) == {"sft_raft", "synthetic"}


def test_optional_inputs_are_marked_in_the_human_readable_label() -> None:
    assert Judge.spec.input_label() == "candidates, sft_raft?, synthetic?"
