"""``mine`` — turn friction events into recovery training examples.

Recovery is the residual an SFT policy must learn: an imitator of success cannot
recover once it deviates. For each friction event the recovery-finder agent
locates the span where the assistant diagnosed and fixed the failure; the span
becomes the trained completion and everything before it is masked context.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.dataset import TrainingExample
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import FrictionEvent, RecoverySpan
from frictiontune.domain.ids import candidate_id, example_id
from frictiontune.domain.responses import HarvestedWork
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import (
    clip,
    failure_reason,
    gather_tolerant,
    read_events_by_episode,
    read_tool_schemas,
    render_transcript,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from frictiontune.core.json import JsonObject
    from frictiontune.pipeline.context import StageContext

_RECOVERY_PROMPT = "recovery_finder"
_HARVEST_PROMPT = "harvest"


def _span_bounds(example: TrainingExample) -> tuple[int, int]:
    """Read the recovery span this stage wrote into an example's ``meta``.

    ``meta`` is a JSON object, so the values arrive as ``JsonValue``; only the
    bounds this stage itself recorded are trusted, and anything else reads as an
    empty span (which never overlaps anything).
    """
    start = example.meta.get("recovery_start")
    end = example.meta.get("recovery_end")
    return (
        start if isinstance(start, int) else 0,
        end if isinstance(end, int) else 0,
    )


class MineSettings(StageSettings):
    """Knobs for mining training examples out of episodes."""

    batch_size: int = 32
    max_transcript_chars: int = 100_000
    min_completion_messages: int = 1
    # The harvest pass mines the other half of the signal: the assistant turns
    # that *avoid* the friction we detect, rather than repairing it. It runs on
    # every episode, not just the ones with events, because a clean episode is
    # exactly where good work is least interrupted.
    harvest: bool = True
    harvest_max_spans: int = 3
    # A span is a behaviour, not a stretch of work: the completion is what gets
    # trained, so a long run teaches rambling. The prompt states the bound and
    # the plumbing enforces it.
    harvest_max_turns: int = 3


class Mine(Stage):
    """Mine training examples from episodes: recoveries, and good work."""

    spec = StageSpec(
        name="mine",
        tier=Tier.A,
        requires_llm=True,
        prompts=["recovery_finder", "harvest"],
        roles=[RoleName.RECOVERY_FINDER, RoleName.HARVESTER],
        lead=Lead.CORE,
        roi="High",
        justification=(
            "Mining only the repair after a failure leaves most of a chat log "
            "unused: the work that needed no repair is the larger half of the "
            "signal, and it is what teaches a model to need less steering."
        ),
        evidence=[
            "InversePRM: recoverable steps carry the signal",
            "measured: 2 448 messages → 96 events → 51 recovery candidates → "
            "5 training examples, while 59 % of episodes (563 assistant turns) "
            "were never a candidate at all",
        ],
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
    ) -> tuple[TrainingExample | None, str]:
        """Extract one recovery example, or return ``(None, reason)``.

        The reason is surfaced in the report so a run that yields no candidates
        says *why* (no recovery, no context, short completion) instead of a bare
        skip count that hides a contract bug.
        """
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
        if not span.found:
            return None, "no_recovery"
        if span.recovery_start is None or span.recovery_end is None:
            return None, "span_incomplete"
        last = len(episode.messages) - 1
        start = max(0, min(span.recovery_start, last))
        end = max(start, min(span.recovery_end, last))
        completion = episode.messages[start : end + 1]
        if len(completion) < settings.min_completion_messages:
            return None, "completion_too_short"
        if not any(message.role == "assistant" for message in completion):
            return None, "completion_without_assistant"
        context = episode.messages[:start]
        if not context:
            return None, "empty_context"
        return (
            TrainingExample(
                id=candidate_id(episode.id, index),
                source=episode.id,
                track="recovery",
                context=list(context),
                completion=list(completion),
                tools=schemas,
                meta={
                    "event_id": event.id,
                    "session_id": episode.session_id,
                    "form": event.form,
                    "cause": event.cause,
                    "cost": event.cost,
                    "leaf": event.leaf,
                    "resolution": event.resolution,
                    "confidence": event.confidence,
                    "trigger": span.trigger,
                    "recovery_start": start,
                    "recovery_end": end,
                    "time_created": episode.time_created,
                },
            ),
            "extracted",
        )

    async def _harvest(
        self,
        episode: Episode,
        ctx: StageContext,
        settings: MineSettings,
        schemas: list[JsonObject],
        taken: list[tuple[int, int]],
    ) -> list[TrainingExample]:
        """Mine the focused good work in one episode.

        ``taken`` holds the recovery spans already mined from this episode: a
        span overlapping one of them would duplicate a behaviour we already have,
        so it is dropped here rather than left for the dedup agent to notice.
        """
        role = ctx.config.role(RoleName.HARVESTER)
        taxonomy = ctx.config.taxonomy
        work = await ctx.run.gateway.complete_structured(
            RoleName.HARVESTER,
            role,
            _HARVEST_PROMPT,
            {
                "episode_id": episode.id,
                "max_spans": str(settings.harvest_max_spans),
                "leaf_axes": "\n".join(
                    f"- {leaf}: form={axis.form}, cause={axis.cause}"
                    + (f" — {axis.description}" if axis.description else "")
                    for leaf, axis in sorted(taxonomy.leaves.items())
                ),
                "transcript": clip(
                    render_transcript(episode.messages), settings.max_transcript_chars
                ),
            },
            HarvestedWork,
        )
        last = len(episode.messages) - 1
        examples: list[TrainingExample] = []
        for span in work.spans[: settings.harvest_max_spans]:
            start = max(0, min(span.start, last))
            end = max(start, min(span.end, last))
            if end - start + 1 > settings.harvest_max_turns:
                continue
            completion = episode.messages[start : end + 1]
            if not any(message.role == "assistant" for message in completion):
                continue
            if any(start <= taken_end and taken_start <= end for taken_start, taken_end in taken):
                continue
            context = episode.messages[:start]
            if not context:
                continue
            examples.append(
                TrainingExample(
                    id=example_id(episode.id, "success", len(examples)),
                    source=episode.id,
                    track="success",
                    context=list(context),
                    completion=list(completion),
                    tools=schemas,
                    meta={
                        "session_id": episode.session_id,
                        "behaviour": span.behaviour,
                        "harvest_start": start,
                        "harvest_end": end,
                        "time_created": episode.time_created,
                    },
                )
            )
        return examples

    async def run(self, ctx: StageContext) -> StageResult:
        """Emit recovery and good-work candidates from the corpus's episodes."""
        settings = ctx.settings(self.settings_model)
        grouped = read_events_by_episode(ctx)
        schemas = read_tool_schemas(ctx)
        sink = ctx.jsonl_sink("candidates")
        produced = 0
        skipped = 0
        episodes_seen = 0
        episodes_total = 0
        harvested = 0
        harvest_episodes = 0
        skip_reasons: Counter[str] = Counter()
        batch: list[tuple[Episode, FrictionEvent, int]] = []
        # The recovery span is the identity of an example: two events in one
        # episode that resolve to the same span produce the same context and
        # completion byte for byte.
        seen_spans: set[tuple[str, object, object]] = set()
        # Recovery spans already mined per episode, so the harvest pass does not
        # re-mine a behaviour the recovery example already teaches.
        recovery_spans: dict[str, list[tuple[int, int]]] = {}

        async def _one(
            item: tuple[Episode, FrictionEvent, int],
        ) -> tuple[TrainingExample | None, str]:
            episode, event, index = item
            return await self._extract(episode, event, ctx, settings, schemas, index)

        async def flush() -> None:
            nonlocal produced, skipped
            if not batch:
                return
            results = await gather_tolerant(_one, batch)
            for result in results:
                if isinstance(result, BaseException):
                    # One event whose reply cannot be validated after repair is
                    # that event's skip, not the stage's failure: the rest of the
                    # batch still yields training data, and the reason is
                    # recorded so a run that yields nothing says why.
                    skip_reasons[failure_reason(result)] += 1
                    skipped += 1
                    continue
                example, reason = result
                if example is None:
                    skip_reasons[reason] += 1
                    skipped += 1
                    continue
                span = (
                    example.source,
                    example.meta.get("recovery_start"),
                    example.meta.get("recovery_end"),
                )
                if span in seen_spans:
                    # An episode with several events often has one recovery that
                    # resolves them all. Emitting that example once per event
                    # would fill the pool with literal copies (42% of the pool on
                    # the first real corpus) and buy a judge call for each, so a
                    # repeated span is dropped here and counted.
                    skip_reasons["duplicate_span"] += 1
                    skipped += 1
                    continue
                seen_spans.add(span)
                recovery_spans.setdefault(example.source, []).append(_span_bounds(example))
                sink.write(example)
                produced += 1
            batch.clear()

        async def harvest_one(episode: Episode) -> list[TrainingExample]:
            nonlocal harvested, harvest_episodes
            try:
                examples = await self._harvest(
                    episode, ctx, settings, schemas, recovery_spans.get(episode.id, [])
                )
            except BaseException as exc:  # one episode's loss, not the stage's
                skip_reasons[f"harvest_{failure_reason(exc)}"] += 1
                return []
            if examples:
                harvest_episodes += 1
            for example in examples:
                sink.write(example)
            harvested += len(examples)
            return examples

        def episodes() -> Iterator[Episode]:
            nonlocal episodes_seen
            for episode in ctx.catalog.stream("episodes", Episode):
                if episode.id in grouped:
                    episodes_seen += 1
                yield episode

        # Recovery first, for every episode that has events, so the harvest pass
        # can see which spans are already taken.
        for episode in episodes():
            for index, event in enumerate(grouped.get(episode.id, [])):
                batch.append((episode, event, index))
                if len(batch) >= settings.batch_size:
                    await flush()
        await flush()

        if settings.harvest:
            harvested_batch: list[Episode] = []
            for episode in ctx.catalog.stream("episodes", Episode):
                episodes_total += 1
                harvested_batch.append(episode)
                if len(harvested_batch) >= settings.batch_size:
                    await gather_tolerant(harvest_one, harvested_batch)
                    harvested_batch.clear()
            if harvested_batch:
                await gather_tolerant(harvest_one, harvested_batch)

        sink.close()

        report = {
            "episodes_with_events": episodes_seen,
            "candidates": produced,
            "skipped": skipped,
            "skip_reasons": dict(skip_reasons),
            "harvest": {
                "enabled": settings.harvest,
                "episodes": episodes_total,
                "episodes_with_spans": harvest_episodes,
                "examples": harvested,
            },
        }
        ctx.emit_json("mine_report", report)
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(skip_reasons.items()))
        ctx.emit_text(
            "report",
            "# Mine\n\n"
            f"- episodes with events: {episodes_seen}\n"
            f"- recovery candidates: {produced}\n"
            f"- skipped: {skipped} ({reasons})\n"
            f"- good-work examples: {harvested} "
            f"(from {harvest_episodes} of {episodes_total} episodes)\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "episodes_with_events": float(episodes_seen),
                "candidates": float(produced),
                "harvested": float(harvested),
                "harvest_episodes": float(harvest_episodes),
                "skipped": float(skipped),
            },
        )
