"""``mine`` — turn friction events into recovery training examples.

Recovery is the residual an SFT policy must learn: an imitator of success cannot
recover once it deviates. For each friction event the recovery-finder agent
locates the span where the assistant diagnosed and fixed the failure; the span
becomes the trained completion and everything before it is masked context.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.dataset import TrainingExample
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import FrictionEvent, RecoverySpan
from frictiontune.domain.ids import candidate_id
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import (
    clip,
    read_events_by_episode,
    read_tool_schemas,
    render_transcript,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from frictiontune.core.json import JsonObject
    from frictiontune.pipeline.context import StageContext

_RECOVERY_PROMPT = "recovery_finder"


class MineSettings(StageSettings):
    """Knobs for recovery extraction."""

    batch_size: int = 32
    max_transcript_chars: int = 100_000
    min_completion_messages: int = 1
    min_confidence: float = 0.0


class Mine(Stage):
    """Extract recovery examples from episodes that contain friction."""

    spec = StageSpec(
        name="mine",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="High",
        justification="Recovery is the residual SFT must learn; success imitation cannot.",
        evidence=["InversePRM: recoverable steps carry the signal"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="low"),
        inputs={"friction_events": "jsonl", "episodes": "jsonl", "tool_schemas": "jsonl"},
        outputs={"candidates": "jsonl", "mine_report": "json"},
        kill_criterion="Drop if self-recovery spans are too rare to matter (measure first).",
    )
    settings_model = MineSettings

    async def _extract(
        self,
        episode: Episode,
        event: FrictionEvent,
        ctx: StageContext,
        settings: MineSettings,
        schemas: list[JsonObject],
        index: int,
    ) -> TrainingExample | None:
        role = ctx.config.role(RoleName.RECOVERY_FINDER)
        span = await ctx.run.gateway.complete_structured(
            RoleName.RECOVERY_FINDER,
            role,
            _RECOVERY_PROMPT,
            {
                "episode_id": episode.id,
                "friction_summary": event.summary or f"{event.form}/{event.leaf}",
                "transcript": clip(
                    render_transcript(episode.messages), settings.max_transcript_chars
                ),
            },
            RecoverySpan,
        )
        if not span.found or span.recovery_start is None or span.recovery_end is None:
            return None
        last = len(episode.messages) - 1
        start = max(0, min(span.recovery_start, last))
        end = max(start, min(span.recovery_end, last))
        completion = episode.messages[start : end + 1]
        if len(completion) < settings.min_completion_messages:
            return None
        if not any(message.role == "assistant" for message in completion):
            return None
        context = episode.messages[:start]
        if not context:
            return None
        return TrainingExample(
            id=candidate_id(episode.id, index),
            source=episode.id,
            track="recovery",
            context=list(context),
            completion=list(completion),
            tools=schemas,
            meta={
                "event_id": event.id,
                "form": event.form,
                "cause": event.cause,
                "cost": event.cost,
                "leaf": event.leaf,
                "resolution": event.resolution,
                "confidence": event.confidence,
                "trigger": span.trigger,
                "failure_turn": span.failure_turn,
                "recovery_start": start,
                "recovery_end": end,
                "time_created": episode.time_created,
            },
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Emit recovery candidates from episodes that carry friction events."""
        settings = ctx.settings(self.settings_model)
        grouped = read_events_by_episode(ctx)
        schemas = read_tool_schemas(ctx)
        sink = ctx.jsonl_sink("candidates")
        produced = 0
        skipped = 0
        episodes_seen = 0
        batch: list[tuple[Episode, FrictionEvent, int]] = []

        async def flush() -> None:
            nonlocal produced, skipped
            if not batch:
                return
            jobs = [
                self._extract(episode, event, ctx, settings, schemas, index)
                for episode, event, index in batch
            ]
            results = await asyncio.gather(*jobs)
            for result in results:
                if isinstance(result, TrainingExample):
                    sink.write(result)
                    produced += 1
                else:
                    skipped += 1
            batch.clear()

        def episodes() -> Iterator[Episode]:
            nonlocal episodes_seen
            for episode in ctx.catalog.stream("episodes", Episode):
                if episode.id in grouped:
                    episodes_seen += 1
                    yield episode

        for episode in episodes():
            for index, event in enumerate(grouped[episode.id]):
                batch.append((episode, event, index))
                if len(batch) >= settings.batch_size:
                    await flush()
        await flush()
        sink.close()

        report = {
            "episodes_with_events": episodes_seen,
            "candidates": produced,
            "skipped": skipped,
        }
        ctx.emit_json("mine_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Mine\n\n"
            f"- episodes with events: {episodes_seen}\n"
            f"- recovery candidates: {produced}\n"
            f"- skipped: {skipped}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "episodes_with_events": float(episodes_seen),
                "candidates": float(produced),
                "skipped": float(skipped),
            },
        )
