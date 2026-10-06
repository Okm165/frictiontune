"""``segment`` — cut sessions into user-bounded episodes and LLM-labelled phases.

The episode (everything between two human messages) is the unit of friction and
of evaluation. Measured corpora show ~22 assistant turns per user turn (p90 59,
max 102), so a long episode is further divided into coherent phases by an agent;
short episodes get a single phase without an LLM call.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.episodes import Episode, EpisodeFeatures, EpisodePhases, Phase, PhaseDraft
from frictiontune.domain.ids import episode_id, phase_id
from frictiontune.domain.trace import Message, ToolPart
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import bounded_gather, clip, render_transcript

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from frictiontune.pipeline.context import StageContext

_PHASES_PROMPT = "segment_phases"


class SegmentSettings(StageSettings):
    """Knobs for episode cutting and phase segmentation."""

    min_turns_for_phases: int = 4
    max_phase_transcript_chars: int = 60_000


def compute_features(messages: Sequence[Message]) -> EpisodeFeatures:
    """Compute descriptive, non-judgmental episode features."""
    turns = tool_calls = tool_errors = reasoning = text_turns = compactions = chars = 0
    for message in messages:
        chars += len(message.content_text())
        if message.role == "assistant":
            turns += 1
        for part in message.parts:
            if isinstance(part, ToolPart):
                tool_calls += 1
                tool_errors += int(part.is_error)
            elif part.kind == "reasoning":
                reasoning += 1
            elif part.kind == "text":
                text_turns += 1
            elif part.kind == "compaction":
                compactions += 1
    return EpisodeFeatures(
        turns=turns,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        reasoning_turns=reasoning,
        text_turns=text_turns,
        compactions=compactions,
        char_count=chars,
    )


def cut_episodes(session_id: str, messages: Sequence[Message]) -> list[Episode]:
    """Split one session's messages into episodes at user-message boundaries."""
    episodes: list[Episode] = []
    current: list[Message] = []
    start_id: str | None = None
    start_seq = 0

    def close(end_seq: int) -> None:
        if start_id is None:
            return
        user_text = next(
            (message.content_text() for message in current if message.role == "user"),
            "",
        )
        start_message = next(
            (message for message in current if message.id == start_id),
            current[0],
        )
        episodes.append(
            Episode(
                id=episode_id(session_id, start_id),
                session_id=session_id,
                start_user_message_id=start_id,
                start_seq=start_seq,
                end_seq=end_seq,
                time_created=start_message.time_created,
                user_text=user_text,
                messages=list(current),
                features=compute_features(current),
            )
        )

    for message in messages:
        if message.role == "user":
            if current:
                close(message.seq)
            current = []
            start_id = message.id
            start_seq = message.seq
        current.append(message)
    if current:
        close(current[-1].seq)
    return episodes


def _sanitize_phases(drafts: Sequence[PhaseDraft], count: int) -> list[PhaseDraft]:
    """Clamp phase drafts to valid, ordered spans; fall back to one phase."""
    if count <= 0:
        return []
    cleaned: list[PhaseDraft] = []
    for draft in drafts:
        start = max(0, min(draft.start_turn, count - 1))
        end = max(start, min(draft.end_turn, count - 1))
        cleaned.append(PhaseDraft(kind=draft.kind, start_turn=start, end_turn=end, goal=draft.goal))
    return cleaned or [PhaseDraft(kind="execute", start_turn=0, end_turn=count - 1)]


class Segment(Stage):
    """Cut messages into episodes and label their phases."""

    spec = StageSpec(
        name="segment",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="High",
        justification=(
            "The user-bounded episode is the unit of friction and evaluation; long "
            "episodes need coherent sub-phases to fit a context window."
        ),
        evidence=["22 assistant turns/user turn (p90 59, max 102)"],
        cost=CostProfile(compute="mixed", api_usd="low", maintenance="low"),
        inputs={"messages": "jsonl"},
        outputs={"episodes": "jsonl", "phases": "jsonl", "segment_report": "json"},
        kill_criterion="Static user-turn cutting is enough only if episodes stay tiny.",
    )
    settings_model = SegmentSettings

    async def _label_phases(
        self, episode: Episode, ctx: StageContext, settings: SegmentSettings
    ) -> list[Phase]:
        assistant_turns = episode.assistant_turns()
        count = len(episode.messages)
        if len(assistant_turns) < settings.min_turns_for_phases:
            drafts = [PhaseDraft(kind="execute", start_turn=0, end_turn=count - 1)]
        else:
            role = ctx.config.role(RoleName.FRICTION_ANALYST)
            result = await ctx.run.gateway.complete_structured(
                RoleName.FRICTION_ANALYST,
                role,
                _PHASES_PROMPT,
                {
                    "episode_id": episode.id,
                    "session_id": episode.session_id,
                    "turn_count": len(assistant_turns),
                    "transcript": clip(
                        render_transcript(episode.messages),
                        settings.max_phase_transcript_chars,
                    ),
                },
                EpisodePhases,
            )
            drafts = _sanitize_phases(result.phases, count)

        return [
            Phase(
                id=phase_id(episode.id, index),
                episode_id=episode.id,
                kind=draft.kind,
                start_turn=draft.start_turn,
                end_turn=draft.end_turn,
                goal=draft.goal,
            )
            for index, draft in enumerate(drafts)
        ]

    def _group_messages(self, ctx: StageContext) -> Iterator[tuple[str, list[Message]]]:
        stream = ctx.catalog.stream("messages", Message)
        for session_id, group in groupby(stream, key=lambda message: message.session_id):
            yield session_id, list(group)

    async def run(self, ctx: StageContext) -> StageResult:
        """Emit episodes (with phases) and a flat phase table."""
        settings = ctx.settings(self.settings_model)
        episode_sink = ctx.jsonl_sink("episodes")
        phase_sink = ctx.jsonl_sink("phases")
        session_count = episode_count = phase_count = 0

        for session_id, messages in self._group_messages(ctx):
            session_count += 1
            episodes = cut_episodes(session_id, messages)
            labelled: list[list[Phase]] = await bounded_gather(  # type: ignore[assignment]
                lambda episode: self._label_phases(episode, ctx, settings),
                episodes,
                limit=settings_min_concurrency(ctx),
            )
            for episode, phases in zip(episodes, labelled, strict=True):
                episode.phases = list(phases)
                episode_sink.write(episode)
                for phase in phases:
                    phase_sink.write(phase)
                    phase_count += 1
                episode_count += 1

        episode_sink.close()
        phase_sink.close()

        report = {
            "sessions": session_count,
            "episodes": episode_count,
            "phases": phase_count,
            "mean_phases_per_episode": (phase_count / episode_count) if episode_count else 0.0,
        }
        ctx.emit_json("segment_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Segment\n\n"
            f"- sessions: {session_count}\n"
            f"- episodes: {episode_count}\n"
            f"- phases: {phase_count}\n"
            f"- mean phases/episode: {report['mean_phases_per_episode']:.2f}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "sessions": float(session_count),
                "episodes": float(episode_count),
                "phases": float(phase_count),
            },
        )


def settings_min_concurrency(ctx: StageContext) -> int:
    """Concurrency cap derived from the friction-analyst role config."""
    return max(1, ctx.config.role(RoleName.FRICTION_ANALYST).concurrency)


__all__ = ["Segment", "SegmentSettings", "compute_features", "cut_episodes"]
