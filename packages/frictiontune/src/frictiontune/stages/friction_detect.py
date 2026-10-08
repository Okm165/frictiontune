"""``friction_detect`` — the signal the whole pipeline is built around.

A cheap router first decides whether an episode merits deep analysis; the
friction-analyst agent then reads the episode and returns typed events with
verbatim evidence. Configured-temperature analysts are run as a small ensemble
and merged by vote, which trades a little cost for calibrated precision.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.episodes import Episode, RouterDecision
from frictiontune.domain.friction import (
    FrictionAnalysis,
    FrictionEvent,
    FrictionEventDraft,
    FrictionProfile,
)
from frictiontune.domain.ids import event_id
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.reporting.report import render_friction_report
from frictiontune.stages.common import (
    analyst_prompt_values,
    bounded_gather,
    clip,
    render_transcript,
    validate_event,
)

if TYPE_CHECKING:
    from frictiontune.pipeline.context import StageContext

_ROUTER_PROMPT = "router"
_ANALYST_PROMPT = "friction_analyst"


class FrictionDetectSettings(StageSettings):
    """Knobs for routing, analysis, and ensemble merging."""

    router_enabled: bool = True
    ensemble: int = 2
    batch_size: int = 32
    max_transcript_chars: int = 100_000
    taxonomy_enforcement: bool = True


@dataclass(slots=True)
class _EpisodeOutcome:
    """Internal per-episode analysis result."""

    episode_id: str
    events: list[FrictionEvent] = field(default_factory=list)
    routed_clean: bool = False
    abstained: bool = False
    invalid: int = 0


def _event_key(draft: FrictionEventDraft) -> tuple[str, tuple[int, ...]]:
    turns = tuple(sorted(span.turn for span in draft.evidence))
    return (draft.leaf, turns)


def _merge_ensemble(
    drafts: list[FrictionEventDraft], *, threshold: int
) -> list[FrictionEventDraft]:
    """Merge draft events across passes, keeping those that reached ``threshold``."""
    grouped: dict[tuple[str, tuple[int, ...]], list[FrictionEventDraft]] = {}
    for draft in drafts:
        grouped.setdefault(_event_key(draft), []).append(draft)
    merged: list[FrictionEventDraft] = []
    for group in grouped.values():
        if len(group) < threshold:
            continue
        best = max(group, key=lambda item: item.confidence)
        mean_confidence = sum(item.confidence for item in group) / len(group)
        merged.append(
            FrictionEventDraft(
                form=best.form,
                cause=best.cause,
                cost=best.cost,
                resolution=best.resolution,
                leaf=best.leaf,
                summary=best.summary,
                evidence=best.evidence,
                confidence=mean_confidence,
            )
        )
    return merged


class FrictionDetect(Stage):
    """Route, analyse, and profile friction across episodes."""

    spec = StageSpec(
        name="friction_detect",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="High",
        justification="The signal: no labels means no rules, no data, and no metric.",
        evidence=["SWE-chat axes; evidence-cited events"],
        cost=CostProfile(compute="llm", api_usd="high", maintenance="medium"),
        inputs={"episodes": "jsonl"},
        outputs={"friction_events": "jsonl", "friction_profile": "json"},
        kill_criterion="Drop the router only if routing cost exceeds the analysis it saves.",
    )
    settings_model = FrictionDetectSettings

    async def _route(
        self, episode: Episode, ctx: StageContext, settings: FrictionDetectSettings
    ) -> bool:
        """Return True if the episode should be analysed in depth."""
        if not settings.router_enabled:
            return True
        role = ctx.config.role(RoleName.ROUTER)
        decision = await ctx.run.gateway.complete_structured(
            RoleName.ROUTER,
            role,
            _ROUTER_PROMPT,
            {
                "session_id": episode.session_id,
                "episode_id": episode.id,
                # The transcript numbers every message; report its length so the
                # metadata matches the indices the model can cite.
                "turn_count": len(episode.messages),
                "tool_calls": episode.features.tool_calls,
                "tool_errors": episode.features.tool_errors,
                "user_turns": episode.features.user_turns_in_session,
                "transcript": clip(
                    render_transcript(episode.messages), settings.max_transcript_chars
                ),
            },
            RouterDecision,
        )
        return decision.decision != "clearly_clean"

    async def _analyse(
        self, episode: Episode, ctx: StageContext, settings: FrictionDetectSettings
    ) -> tuple[list[FrictionEventDraft], bool, int]:
        """Run the analyst (with optional ensemble); return drafts, abstain, passes."""
        role = ctx.config.role(RoleName.FRICTION_ANALYST)
        values = analyst_prompt_values(
            ctx.config,
            session_id=episode.session_id,
            episode_id=episode.id,
            turn_count=len(episode.messages),
            tool_calls=episode.features.tool_calls,
            tool_errors=episode.features.tool_errors,
            compactions=episode.features.compactions,
            transcript=render_transcript(episode.messages),
            max_chars=settings.max_transcript_chars,
        )

        passes = settings.ensemble if role.temperature > 0.0 else 1
        if passes <= 1:
            analysis = await ctx.run.gateway.complete_structured(
                RoleName.FRICTION_ANALYST, role, _ANALYST_PROMPT, values, FrictionAnalysis
            )
            return list(analysis.events), analysis.abstain, 1

        sampling = role.model_copy(update={"cache": False})
        results = await asyncio.gather(
            *(
                ctx.run.gateway.complete_structured(
                    RoleName.FRICTION_ANALYST, sampling, _ANALYST_PROMPT, values, FrictionAnalysis
                )
                for _ in range(passes)
            )
        )
        drafts = [draft for analysis in results for draft in analysis.events]
        threshold = max(1, round(passes / 2))
        return _merge_ensemble(drafts, threshold=threshold), any(r.abstain for r in results), passes

    async def _process(
        self, episode: Episode, ctx: StageContext, settings: FrictionDetectSettings
    ) -> _EpisodeOutcome:
        if not await self._route(episode, ctx, settings):
            return _EpisodeOutcome(episode_id=episode.id, routed_clean=True)

        drafts, abstained, passes = await self._analyse(episode, ctx, settings)
        events: list[FrictionEvent] = []
        invalid = 0
        for draft in drafts:
            event = FrictionEvent(
                id=event_id(episode.id, len(events)),
                episode_id=episode.id,
                ensemble_votes=passes,
                **draft.model_dump(),
            )
            if settings.taxonomy_enforcement:
                try:
                    event = validate_event(event, ctx.config)
                except Exception:
                    invalid += 1
                    ctx.logger.warning(
                        "dropped friction event with invalid taxonomy",
                        episode_id=episode.id,
                        leaf=event.leaf,
                    )
                    continue
            events.append(event)
        return _EpisodeOutcome(
            episode_id=episode.id, events=events, abstained=abstained, invalid=invalid
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Analyse every episode and emit events plus an aggregate profile."""
        settings = ctx.settings(self.settings_model)
        profile = FrictionProfile()
        sink = ctx.jsonl_sink("friction_events")
        total_invalid = 0
        concurrency = ctx.config.role(RoleName.FRICTION_ANALYST).concurrency

        def absorb(outcome: _EpisodeOutcome) -> None:
            nonlocal total_invalid
            profile.episodes_analysed += 1
            if outcome.routed_clean:
                return
            if outcome.abstained:
                profile.abstentions += 1
            if outcome.events:
                profile.episodes_with_friction += 1
            profile.total_events += len(outcome.events)
            total_invalid += outcome.invalid
            for event in outcome.events:
                sink.write(event)
                profile.by_form[event.form] = profile.by_form.get(event.form, 0) + 1
                profile.by_cause[event.cause] = profile.by_cause.get(event.cause, 0) + 1
                profile.by_leaf[event.leaf] = profile.by_leaf.get(event.leaf, 0) + 1
                profile.by_resolution[event.resolution] = (
                    profile.by_resolution.get(event.resolution, 0) + 1
                )

        batch: list[Episode] = []

        async def flush() -> None:
            if not batch:
                return
            outcomes: list[_EpisodeOutcome] = await bounded_gather(  # type: ignore[assignment]
                lambda episode: self._process(episode, ctx, settings),
                batch,
                limit=concurrency,
            )
            for outcome in outcomes:
                absorb(outcome)
            batch.clear()

        for episode in ctx.catalog.stream("episodes", Episode):
            batch.append(episode)
            if len(batch) >= settings.batch_size:
                await flush()
        await flush()
        sink.close()

        ctx.emit_json("friction_profile.json", profile)
        ctx.emit_text("report.md", render_friction_report(profile))
        return StageResult.from_context(
            ctx,
            metrics={
                "episodes": float(profile.episodes_analysed),
                "episodes_with_friction": float(profile.episodes_with_friction),
                "events": float(profile.total_events),
                "coverage": profile.coverage(),
                "abstentions": float(profile.abstentions),
                "invalid_events": float(total_invalid),
                "mean_events_per_episode": (
                    profile.total_events / profile.episodes_analysed
                    if profile.episodes_analysed
                    else 0.0
                ),
            },
        )
