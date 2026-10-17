"""``export`` — merge or package the adapter for serving, with a parity check.

The final step is intentionally small: delegate to the serving backend, then
record the parity result so a silent export regression cannot pass unnoticed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from frictiontune.backends.models import ExportRequest, ExportResult, ProbeExample
from frictiontune.core.errors import StageError
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

if TYPE_CHECKING:
    from frictiontune.core.json import JsonObject
    from frictiontune.pipeline.context import StageContext


class ExportSettings(StageSettings):
    """Knobs for merging/packaging and the parity probe."""

    method: str = "merge"
    probe_prompts: list[str] = Field(default_factory=list)


class Export(Stage):
    """Export the trained adapter for serving."""

    spec = StageSpec(
        name="export",
        tier=Tier.A,
        lead=Lead.CORE,
        roi="High",
        justification="Serving the adapter is trivial effort but blocks any downstream use.",
        evidence=["merge or vLLM --enable-lora; parity check"],
        cost=CostProfile(compute="cpu", api_usd="none", maintenance="low"),
        inputs={"adapter": "json"},
        outputs={"export_result": "json"},
        kill_criterion="None; exporting is required to use the adapter.",
    )
    settings_model = ExportSettings

    async def run(self, ctx: StageContext) -> StageResult:
        """Delegate the export and record its parity outcome."""
        settings = ctx.settings(self.settings_model)
        raw = ctx.catalog.read_json("adapter")
        if not isinstance(raw, dict):
            raise StageError("export", "adapter artifact is not a JSON object")
        adapter: JsonObject = raw
        base_model = str(adapter.get("base_model", ""))
        adapter_dir = str(adapter.get("adapter_dir", ""))
        if not base_model or not adapter_dir:
            raise StageError("export", "adapter artifact is missing base_model/adapter_dir")

        request = ExportRequest(
            base_model=base_model,
            adapter_dir=adapter_dir,
            output_dir=str(ctx.store.directory / "served"),
            method="lora" if settings.method == "lora" else "merge",
            probe=[ProbeExample(prompt=prompt) for prompt in settings.probe_prompts],
        )
        result: ExportResult = await ctx.run.backends.serving().export(request)
        ctx.emit_json("export_result", result)
        lines = [
            "# Export",
            "",
            f"- method: {request.method}",
            f"- served dir: `{result.served_dir}`",
            f"- merged: {result.merged}",
            f"- parity: {result.parity.matches}/{result.parity.total}",
        ]
        if result.parity.mismatches:
            # A count alone cannot be read: a merge that rounded in bf16 and a
            # merge that is broken both show up here, and only the outputs say
            # which one happened.
            lines += ["", "## Probes that differed", ""]
            for mismatch in result.parity.mismatches:
                lines += [
                    f"- prompt: {mismatch.prompt!r}",
                    f"  - adapter: {mismatch.expected!r}",
                    f"  - merged:  {mismatch.actual!r}",
                ]
        ctx.emit_text("report", "\n".join(lines) + "\n")
        return StageResult.from_context(
            ctx,
            metrics={
                "merged": float(result.merged),
                "parity_matches": float(result.parity.matches),
                "parity_total": float(result.parity.total),
            },
            notes=[] if result.parity.ok else ["parity probe did not fully match"],
        )
