"""The orchestrator: runs stages in dependency order with idempotency.

A stage invocation is idempotent by construction: its run id is a hash of
*(stage, stage config, code revision, input hashes)*. If a matching manifest
already exists it is reused, its outputs are re-published into the catalog, and
its work is skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from frictiontune.core.errors import GateError, StageError
from frictiontune.core.hashing import compute_run_id, content_hash
from frictiontune.pipeline.context import RunContext, StageContext
from frictiontune.pipeline.manifest import (
    RunManifest,
    StageManifest,
    code_revision,
    read_stage_manifest,
    utc_now,
    write_run_manifest,
    write_stage_manifest,
)
from frictiontune.pipeline.spec import Tier
from frictiontune.version import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.core.artifacts import DatasetRef
    from frictiontune.pipeline.dag import DependencyGraph
    from frictiontune.pipeline.registry import StageRegistry


def _kind_for(ref: DatasetRef) -> str:
    suffix = Path(ref.path).suffix
    return {
        ".jsonl": "jsonl",
        ".json": "json",
        ".md": "text",
        ".txt": "text",
    }.get(suffix, "text")


@dataclass(slots=True)
class Orchestrator:
    """Executes a set of stages in dependency order."""

    run: RunContext
    registry: StageRegistry
    graph: DependencyGraph
    force: bool = False
    allow_gated: bool = False

    async def execute(
        self,
        stages: Sequence[str] | None = None,
        *,
        include_dependencies: bool = False,
        cached_dependencies_only: bool = False,
    ) -> RunManifest:
        """Run ``stages`` (or the core set) and return the run manifest.

        ``cached_dependencies_only`` refuses to *start* a stage that was not
        asked for by name: every upstream stage must already have a manifest.
        Phases that cannot host the model — training stops the llama.cpp servers
        to free the GPUs for the job — set it, so a cold cache stops with the
        stage's name instead of re-running an LLM stage against an endpoint that
        is not there and retrying connection errors for as long as it is allowed
        to.
        """
        if stages is None:
            selected = [stage.name for stage in self.registry.core()]
        else:
            selected = list(stages)

        requested = set(selected)
        order = self.graph.closure(selected) if include_dependencies else self.graph.order(selected)
        started = utc_now()
        revision = code_revision()
        entries: list[StageManifest] = []

        for name in order:
            entries.append(
                await self._run_stage(
                    name,
                    revision,
                    require_cached=cached_dependencies_only and name not in requested,
                )
            )

        finished = utc_now()
        manifest = RunManifest(
            pipeline=self.run.pipeline,
            project=self.run.config.project,
            code_revision=revision,
            config_fingerprint=content_hash(self.run.config.model_dump(mode="json")),
            artifacts_root=str(self.run.artifacts_root),
            started_at=started,
            finished_at=finished,
            stages=entries,
        )
        write_run_manifest(self.run.artifacts_root / self.run.pipeline / "manifest.json", manifest)
        return manifest

    async def _run_stage(
        self, name: str, revision: str, *, require_cached: bool = False
    ) -> StageManifest:
        stage = self.registry.get(name)
        spec = stage.spec

        if spec.tier is Tier.C and not self.allow_gated:
            raise GateError(
                f"stage {name!r} is Tier C (gated by {spec.gated_by!r}); pass allow_gated to run it"
            )
        if spec.tier is Tier.D:
            raise GateError(f"stage {name!r} is Tier D and must never run")

        missing = [key for key in spec.inputs if not self.run.catalog.has(key)]
        if missing:
            raise StageError(name, f"missing input artifact(s): {missing}")

        # Optional inputs are consumed only when present; whether they are present
        # feeds the run id so a stale manifest is never reused after a gated
        # extension has produced (or changed) one of them.
        optional_present = [key for key in spec.optional_inputs if self.run.catalog.has(key)]
        consumed = (*spec.inputs, *optional_present)

        input_refs = {key: self.run.catalog.get(key).ref for key in consumed}
        input_hashes = self.run.catalog.hashes(consumed)
        input_hashes.extend(stage.external_input_hashes(self.run))

        # The run id must change whenever *anything* that affects the stage's output
        # changes: what the stage is configured to do, the models and sampling it
        # is bound to, the prompt library, and the version. It must *not* change
        # when a phase merely points the same roles at the same models through
        # different endpoints, or the stage cache would miss on every phase
        # boundary (see RunConfig.semantic_fingerprint).
        # A stage that calls no model depends on no model binding or prompt: giving
        # it an empty set for both keeps its identity stable across phases. A
        # stage that does call models but names none keeps the conservative
        # whole-library binding.
        declared_roles = spec.roles if spec.roles else (None if spec.requires_llm else [])
        declared_prompts = spec.prompts if spec.prompts else (None if spec.requires_llm else [])
        config_slice = {
            "config": self.run.config.semantic_fingerprint(
                stage=name, roles=declared_roles, uses_evaluation=spec.uses_evaluation
            ),
            "prompts": self.run.gateway.prompt_fingerprint(declared_prompts),
            "version": __version__,
        }
        run_id = compute_run_id(
            stage=name, config=config_slice, code_revision=revision, input_hashes=input_hashes
        )
        store = self.run.store(name, run_id)

        if store.exists("manifest.json") and not self.force:
            manifest = read_stage_manifest(store.path("manifest.json"))
            for ref in manifest.outputs.values():
                self.run.catalog.publish(ref, kind=ref.kind or _kind_for(ref))
            manifest.cached = True
            self.run.logger.info("stage cache hit", stage=name, run_id=run_id)
            return manifest

        if require_cached:
            raise StageError(
                name,
                "not cached, and this run was told to execute only the stages it "
                "was asked for (--cached-deps). Run the upstream phase first; this "
                "phase cannot rebuild the pipeline.",
            )

        ctx = StageContext(run=self.run, stage=name, run_id=run_id, store=store)
        log = ctx.logger.bind(tier=spec.tier.value)
        log.info("stage start", inputs=list(consumed))
        started = utc_now()
        tick = time.monotonic()
        result = await stage.run(ctx)
        undeclared = sorted(set(result.outputs) - set(spec.outputs))
        if undeclared:
            log.warning("stage emitted undeclared output(s)", outputs=undeclared)
        finished = utc_now()
        elapsed = time.monotonic() - tick
        for key, value in result.metrics.items():
            self.run.metrics[f"{name}.{key}"] = value

        manifest = StageManifest(
            stage=name,
            run_id=run_id,
            spec=spec,
            code_revision=revision,
            config_fingerprint=content_hash(config_slice),
            inputs=input_refs,
            outputs=result.outputs,
            metrics=result.metrics,
            notes=result.notes,
            started_at=started,
            finished_at=finished,
            duration_s=elapsed,
        )
        write_stage_manifest(store.path("manifest.json"), manifest)
        log.info("stage done", outputs=list(result.outputs), metrics=result.metrics)
        return manifest
