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
from frictiontune.core.errors import StageError
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
    clip,
    gather_tolerant,
    render_transcript,
    validate_event,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import structlog

    from frictiontune.pipeline.context import PublishedSink, StageContext

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


def absorb_batch(
    batch: Sequence[Episode],
    results: Sequence[_EpisodeOutcome | BaseException],
    profile: FrictionProfile,
    sink: PublishedSink,
    logger: structlog.stdlib.BoundLogger,
) -> int:
    """Fold one batch's outcomes into ``profile``; return the invalid-event count.

    A failed episode is *that episode's* outcome, not the batch's. The analyst's
    ensemble refuses to decide on a quorum it does not have (``_healthy_passes``),
    so such an episode contributes no events — but it is counted as ``unanalysed``
    rather than allowed to take the stage down with it. Counting it separately
    keeps ``coverage()`` a statement about the episodes that *were* analysed.
    """
    invalid = 0
    for episode, result in zip(batch, results, strict=True):
        if isinstance(result, BaseException):
            profile.unanalysed += 1
            logger.warning(
                "episode could not be analysed",
                episode_id=episode.id,
                reason=type(result).__name__,
                error=str(result),
            )
            continue
        profile.episodes_analysed += 1
        invalid += result.invalid
        if result.routed_clean:
            continue
        if result.abstained:
            profile.abstentions += 1
        if result.events:
            profile.episodes_with_friction += 1
        profile.total_events += len(result.events)
        for event in result.events:
            sink.write(event)
            profile.by_form[event.form] = profile.by_form.get(event.form, 0) + 1
            profile.by_cause[event.cause] = profile.by_cause.get(event.cause, 0) + 1
            profile.by_leaf[event.leaf] = profile.by_leaf.get(event.leaf, 0) + 1
            profile.by_resolution[event.resolution] = (
                profile.by_resolution.get(event.resolution, 0) + 1
            )
    return invalid


def _evidence_overlap(a: FrictionEventDraft, b: FrictionEventDraft) -> bool:
    """Whether two drafts cite at least one common transcript turn."""
    return bool({span.turn for span in a.evidence} & {span.turn for span in b.evidence})


def _vote_threshold(passes: int) -> int:
    """Strict majority of independent passes (2 of 2, 2 of 3, 3 of 4)."""
    return passes // 2 + 1


def _healthy_passes(
    results: Sequence[FrictionAnalysis | BaseException], *, passes: int
) -> list[FrictionAnalysis]:
    """Keep the ensemble passes that completed; raise only when too few survive.

    A single transient failure must not abort the whole stage, but nor may it
    silently lower the bar: consensus still needs ``_vote_threshold(passes)``
    independent passes, so when fewer survive we surface the failure instead of
    deciding on a quorum we do not have.
    """
    healthy = [result for result in results if isinstance(result, FrictionAnalysis)]
    if len(healthy) < _vote_threshold(passes):
        failure = next((result for result in results if isinstance(result, BaseException)), None)
        if failure is not None:
            raise failure
    return healthy


def _merge_ensemble(
    passes: list[list[FrictionEventDraft]], *, threshold: int
) -> list[tuple[FrictionEventDraft, int]]:
    """Merge per-pass drafts into consensus events and count their votes.

    A consensus event is a cluster of drafts that share a leaf and cite
    overlapping evidence turns. Overlap-clustering (rather than exact turn-set
    equality) keeps the same friction found at slightly different turn indices
    from being double-counted, while distinct events that share a leaf but not
    their evidence stay separate. A cluster survives only when at least
    ``threshold`` distinct passes support it; the returned vote count is the
    number of supporting passes, not merely the number of drafts.
    """
    clusters: list[list[tuple[int, FrictionEventDraft]]] = []
    for pass_index, drafts in enumerate(passes):
        for draft in drafts:
            cluster = next(
                (
                    candidate
                    for candidate in clusters
                    if candidate[0][1].leaf == draft.leaf
                    and any(_evidence_overlap(draft, member[1]) for member in candidate)
                ),
                None,
            )
            if cluster is None:
                clusters.append([(pass_index, draft)])
            else:
                cluster.append((pass_index, draft))

    merged: list[tuple[FrictionEventDraft, int]] = []
    for cluster in clusters:
        votes = len({pass_index for pass_index, _draft in cluster})
        if votes < threshold:
            continue
        best = max((draft for _pass, draft in cluster), key=lambda item: item.confidence)
        # Round the averaged probability so artifacts do not carry float noise
        # (e.g. 0.5900000000000001); 4 decimals is finer than the model's
        # calibration and keeps the JSON readable.
        mean_confidence = round(sum(draft.confidence for _pass, draft in cluster) / len(cluster), 4)
        merged.append(
            (
                FrictionEventDraft(
                    cost=best.cost,
                    resolution=best.resolution,
                    leaf=best.leaf,
                    summary=best.summary,
                    evidence=best.evidence,
                    confidence=mean_confidence,
                ),
                votes,
            )
        )
    return merged


class FrictionDetect(Stage):
    """Route, analyse, and profile friction across episodes."""

    spec = StageSpec(
        name="friction_detect",
        tier=Tier.A,
        requires_llm=True,
        prompts=["router", "friction_analyst"],
        roles=[RoleName.ROUTER, RoleName.FRICTION_ANALYST],
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
    ) -> tuple[list[tuple[FrictionEventDraft, int]], bool]:
        """Run the analyst (with optional ensemble); return drafts with votes, abstain."""
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
            return [(event, 1) for event in analysis.events], analysis.abstain

        sampling = role.model_copy(update={"cache": False})
        calls = [
            ctx.run.gateway.complete_structured(
                RoleName.FRICTION_ANALYST, sampling, _ANALYST_PROMPT, values, FrictionAnalysis
            )
            for _ in range(passes)
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)
        threshold = _vote_threshold(passes)
        healthy = _healthy_passes(results, passes=passes)
        if len(healthy) < passes:
            ctx.logger.warning(
                "analyst ensemble degraded: some passes failed",
                episode_id=episode.id,
                passes=passes,
                succeeded=len(healthy),
            )
        return (
            _merge_ensemble([list(result.events) for result in healthy], threshold=threshold),
            any(result.abstain for result in healthy),
        )

    async def _process(
        self, episode: Episode, ctx: StageContext, settings: FrictionDetectSettings
    ) -> _EpisodeOutcome:
        if not await self._route(episode, ctx, settings):
            return _EpisodeOutcome(episode_id=episode.id, routed_clean=True)

        taxonomy = ctx.config.taxonomy
        drafts, abstained = await self._analyse(episode, ctx, settings)
        events: list[FrictionEvent] = []
        invalid = 0
        for draft, votes in drafts:
            # form/cause are a declared function of the leaf, so the plumbing
            # derives them instead of asking the model to restate the mapping.
            axes = taxonomy.leaves.get(draft.leaf)
            if axes is None and settings.taxonomy_enforcement:
                invalid += 1
                ctx.logger.warning(
                    "dropped friction event with unknown leaf",
                    episode_id=episode.id,
                    leaf=draft.leaf,
                )
                continue
            form, cause = (axes.form, axes.cause) if axes is not None else ("unknown", "unknown")
            event = FrictionEvent(
                id=event_id(episode.id, len(events)),
                episode_id=episode.id,
                ensemble_votes=votes,
                form=form,
                cause=cause,
                **draft.model_dump(),
            )
            if settings.taxonomy_enforcement:
                try:
                    event = validate_event(event, ctx.config)
                except StageError:
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
        batch: list[Episode] = []

        async def flush() -> None:
            nonlocal total_invalid
            if not batch:
                return
            results = await gather_tolerant(
                lambda episode: self._process(episode, ctx, settings), batch
            )
            total_invalid += absorb_batch(batch, results, profile, sink, ctx.logger)
            batch.clear()

        for episode in ctx.catalog.stream("episodes", Episode):
            batch.append(episode)
            if len(batch) >= settings.batch_size:
                await flush()
        await flush()
        sink.close()

        ctx.emit_json("friction_profile", profile)
        ctx.emit_text("report", render_friction_report(profile))
        return StageResult.from_context(
            ctx,
            metrics={
                "episodes": float(profile.episodes_analysed),
                "episodes_with_friction": float(profile.episodes_with_friction),
                "events": float(profile.total_events),
                "coverage": profile.coverage(),
                "abstentions": float(profile.abstentions),
                "unanalysed": float(profile.unanalysed),
                "invalid_events": float(total_invalid),
                "mean_events_per_episode": (
                    profile.total_events / profile.episodes_analysed
                    if profile.episodes_analysed
                    else 0.0
                ),
            },
        )
