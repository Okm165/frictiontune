"""``judge`` — the shared quality gate every candidate must pass.

LIMA's lesson is that unfiltered data actively hurts, so the gate is strict: a
rubric of five 0-5 axes, an explicit accept decision, and a deterministic
post-check that a model cannot over-accept a candidate it scored as
disqualifying on adherence or correctness.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.dataset import TrainingExample
from frictiontune.domain.responses import JudgeVerdict
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from frictiontune.pipeline.context import StageContext

_JUDGE_PROMPT = "judge"

_DISQUALIFYING_SCORE = 2


class JudgeSettings(StageSettings):
    """Knobs for the judging gate."""

    accept_threshold: int = 20
    batch_size: int = 32
    max_context_chars: int = 60_000
    max_completion_chars: int = 20_000


def _accepted(verdict: JudgeVerdict, *, threshold: int) -> bool:
    """Deterministic post-check over the model's verdict."""
    if verdict.task_adherence <= _DISQUALIFYING_SCORE:
        return False
    if verdict.correctness <= _DISQUALIFYING_SCORE:
        return False
    return verdict.accept and verdict.total() >= threshold


class Judge(Stage):
    """Score and filter candidate training examples."""

    spec = StageSpec(
        name="judge",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="High",
        justification="LIMA: unfiltered data hurts; this gate protects every downstream stage.",
        evidence=["LIMA 2305.11206"],
        cost=CostProfile(compute="llm", api_usd="high", maintenance="low"),
        inputs={"candidates": "jsonl"},
        # Gated data tracks (RAFT success examples, synthesized coverage) are
        # folded into the same gate when their producers have run. They are
        # optional so a core run never depends on a Tier C stage.
        optional_inputs={"sft_raft": "jsonl", "synthetic": "jsonl"},
        outputs={"judged": "jsonl", "judge_report": "json"},
        kill_criterion=(
            "Relax the threshold only if training shows under-fitting from too little data."
        ),
    )
    settings_model = JudgeSettings

    async def _judge_one(
        self, candidate: TrainingExample, ctx: StageContext, settings: JudgeSettings
    ) -> tuple[TrainingExample, JudgeVerdict, bool]:
        role = ctx.config.role(RoleName.JUDGE)
        verdict = await ctx.run.gateway.complete_structured(
            RoleName.JUDGE,
            role,
            _JUDGE_PROMPT,
            {
                "candidate_id": candidate.id,
                "source": candidate.source,
                "track": candidate.track,
                "context": clip(render_transcript(candidate.context), settings.max_context_chars),
                "completion": clip(
                    render_transcript(candidate.completion), settings.max_completion_chars
                ),
            },
            JudgeVerdict,
        )
        accepted = _accepted(verdict, threshold=settings.accept_threshold)
        scored = candidate.model_copy(
            update={"judge_score": float(verdict.total()), "kept": accepted}
        )
        return scored, verdict, accepted

    async def run(self, ctx: StageContext) -> StageResult:
        """Score every candidate, keeping only those that pass the gate."""
        settings = ctx.settings(self.settings_model)
        sink = ctx.jsonl_sink("judged")
        judged = kept = 0
        batch: list[TrainingExample] = []

        async def flush() -> None:
            nonlocal judged, kept
            if not batch:
                return
            results = await asyncio.gather(
                *(self._judge_one(candidate, ctx, settings) for candidate in batch)
            )
            for scored, _verdict, accepted in results:
                sink.write(scored)
                judged += 1
                kept += int(accepted)
            batch.clear()

        for source in ("candidates", *(k for k in self.spec.optional_inputs if ctx.catalog.has(k))):
            for candidate in ctx.catalog.stream(source, TrainingExample):
                batch.append(candidate)
                if len(batch) >= settings.batch_size:
                    await flush()
        await flush()
        sink.close()

        report = {"judged": judged, "kept": kept, "rejected": judged - kept}
        ctx.emit_json("judge_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Judge\n\n"
            f"- judged: {judged}\n"
            f"- kept: {kept}\n"
            f"- rejected: {judged - kept}\n"
            f"- accept threshold: {settings.accept_threshold}/25\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "judged": float(judged),
                "kept": float(kept),
                "rejected": float(judged - kept),
                "accept_rate": (kept / judged) if judged else 0.0,
            },
        )
