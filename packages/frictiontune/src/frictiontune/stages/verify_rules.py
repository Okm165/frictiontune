"""``verify_rules`` — keep the rule track safe with agentic replay.

Each rule is replayed against a held-out sample of historical episodes: would it
have fired, would it have prevented the targeted friction, and would it have
blocked correct behaviour (collateral)? Collateral is the failure mode that makes
rules dangerous, so the accept gate requires prevention and forbids collateral.

This stage is the pipeline's largest consumer of LLM calls — every rule is
verified on ``episodes_per_rule + clean_episodes_per_rule`` episodes — so the
prompt's *layout* is a cost decision. The prompt puts the episode id and the
transcript **before** the rule, because the transcript is invariant across the
rules sampled for one episode while the rule varies: llama.cpp's server reuses
the longest common prefix of a slot's previous prompt, so an invariant block
placed first is prefilled once per episode instead of once per (rule, episode).
The rule-first layout this stage shipped with made every call re-prefill the
whole transcript — the dominant cost of the whole data phase.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import FrictionEvent
from frictiontune.domain.responses import RuleVerdict
from frictiontune.domain.rules import RuleLibrary, RuleStatus, RuleVerification, RuntimeRule
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, gather_tolerant, render_transcript

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from frictiontune.pipeline.context import StageContext

_VERIFY_PROMPT = "rule_verifier"


class VerifyRulesSettings(StageSettings):
    """Knobs for the replay verification gate."""

    episodes_per_rule: int = 5
    clean_episodes_per_rule: int = 3
    pool_target: int = 12
    pool_clean: int = 8
    min_prevented: int = 1
    max_collateral: int = 0
    max_transcript_chars: int = 80_000


def _read_events(ctx: StageContext) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Map each episode to a friction summary and to the events it contains.

    The event-to-episode map lets the gate hold out any episode a rule was
    compiled from: replaying a rule on its own source would be circular.
    """
    summaries: dict[str, str] = defaultdict(str)
    event_ids: dict[str, set[str]] = defaultdict(set)
    for event in ctx.catalog.stream("friction_events", FrictionEvent):
        if not summaries[event.episode_id]:
            summaries[event.episode_id] = event.summary or f"{event.form}/{event.leaf}"
        event_ids[event.episode_id].add(event.id)
    return summaries, event_ids


def _episode_sessions(ctx: StageContext) -> dict[str, str]:
    """Map every episode id to the session it belongs to, for the holdout."""
    return {episode.id: episode.session_id for episode in ctx.catalog.stream("episodes", Episode)}


def _held_out(
    pool: Sequence[Episode],
    event_ids: Mapping[str, set[str]],
    event_session: Mapping[str, str],
    source_events: Collection[str],
    limit: int,
) -> list[Episode]:
    """First ``limit`` pool episodes genuinely independent of the rule.

    A rule is compiled from the friction in particular episodes, so replaying it
    on those same episodes would inflate ``prevented`` and let a rule pass its own
    test. Excluding the *session* as well as the source episodes closes the wider
    leak: episodes of one session share the user's intent, the tools, and the
    transcript context, so a same-session episode is not independent evidence and
    would let a rule pass on a near-copy of its own derivation. Clean episodes
    carry no events and are always held out by construction.
    """
    own_sessions = {event_session[event] for event in source_events if event in event_session}
    own_events = set(source_events)
    selected: list[Episode] = []
    for episode in pool:
        if len(selected) >= limit:
            break
        if episode.session_id in own_sessions:
            continue
        if own_events & event_ids.get(episode.id, set()):
            continue
        selected.append(episode)
    return selected


def _sample_pools(
    ctx: StageContext,
    summaries: dict[str, str],
    settings: VerifyRulesSettings,
) -> tuple[list[Episode], list[Episode]]:
    """Collect bounded pools of friction-bearing and clean episodes in stream order."""
    targeted: list[Episode] = []
    clean: list[Episode] = []
    for episode in ctx.catalog.stream("episodes", Episode):
        if episode.id in summaries:
            if len(targeted) < settings.pool_target:
                targeted.append(episode)
        elif len(clean) < settings.pool_clean:
            clean.append(episode)
        if len(targeted) >= settings.pool_target and len(clean) >= settings.pool_clean:
            break
    return targeted, clean


class VerifyRules(Stage):
    """Replay every rule against sampled episodes and gate its activation."""

    spec = StageSpec(
        name="verify_rules",
        tier=Tier.A,
        requires_llm=True,
        prompts=["rule_verifier"],
        roles=[RoleName.RULE_VERIFIER],
        lead=Lead.CORE,
        roi="High",
        justification="Keeps the rule track safe by measuring collateral before activation.",
        evidence=["agentic replay; per-rule held-out sample; collateral is the dominant rule risk"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="low"),
        inputs={"rules": "json", "episodes": "jsonl", "friction_events": "jsonl"},
        outputs={"rules_verified": "json", "verify_rules_report": "json"},
        kill_criterion="Drop if rules never prevent friction without collateral.",
    )
    settings_model = VerifyRulesSettings

    async def _check(
        self,
        rule: RuntimeRule,
        episode: Episode,
        friction_summary: str,
        ctx: StageContext,
        settings: VerifyRulesSettings,
    ) -> RuleVerdict:
        role = ctx.config.role(RoleName.RULE_VERIFIER)
        return await ctx.run.gateway.complete_structured(
            RoleName.RULE_VERIFIER,
            role,
            _VERIFY_PROMPT,
            {
                "applicability": rule.applicability,
                "instruction": rule.instruction,
                "verifier_spec": rule.verifier.spec,
                "episode_id": episode.id,
                "friction_summary": friction_summary or "no recorded friction",
                "transcript": clip(
                    render_transcript(episode.messages), settings.max_transcript_chars
                ),
            },
            RuleVerdict,
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Verify each rule and write the updated library."""
        settings = ctx.settings(self.settings_model)
        raw = ctx.catalog.read_json("rules")
        library = RuleLibrary.model_validate(raw)
        summaries, event_ids = _read_events(ctx)
        episode_session = _episode_sessions(ctx)
        # Flatten event -> session so the holdout can exclude a rule's whole
        # source session, not just the episodes it was compiled from.
        event_session = {
            event: episode_session[episode]
            for episode, events in event_ids.items()
            for event in events
            if episode in episode_session
        }
        targeted, clean = _sample_pools(ctx, summaries, settings)

        verified: list[RuntimeRule] = []
        accepted_count = 0
        held_out_limited = 0
        checks_failed = 0
        for rule in library.rules:
            targeted_episodes = _held_out(
                targeted,
                event_ids,
                event_session,
                rule.source_events,
                settings.episodes_per_rule,
            )
            held_out_limited += int(len(targeted_episodes) < settings.episodes_per_rule)
            episodes = targeted_episodes + clean[: settings.clean_episodes_per_rule]

            async def check_episode(episode: Episode, _rule: RuntimeRule = rule) -> RuleVerdict:
                return await self._check(
                    _rule, episode, summaries.get(episode.id, ""), ctx, settings
                )

            results = await gather_tolerant(check_episode, episodes)
            # A check that fails yields no evidence the rule helps: the episode
            # is not counted as fired, prevented, or collateral. This keeps the
            # gate conservative — a rule whose replay is incomplete can never
            # accumulate the prevention it needs to be activated.
            checks_failed += sum(1 for result in results if isinstance(result, BaseException))
            verdicts = [result for result in results if not isinstance(result, BaseException)]
            fires = sum(1 for verdict in verdicts if verdict.fires)
            prevented = sum(1 for verdict in verdicts if verdict.fires and verdict.prevents)
            collateral = sum(1 for verdict in verdicts if verdict.collateral)
            mean_conf = (
                round(sum(verdict.confidence for verdict in verdicts) / len(verdicts), 4)
                if verdicts
                else 0.0
            )
            verification = RuleVerification(
                episodes_checked=len(verdicts),
                fires=fires,
                prevented=prevented,
                collateral=collateral,
                mean_confidence=mean_conf,
            )
            status = (
                RuleStatus.ACTIVE
                if verification.accepted(
                    min_prevented=settings.min_prevented,
                    max_collateral=settings.max_collateral,
                )
                else RuleStatus.CANDIDATE
            )
            accepted_count += int(status is RuleStatus.ACTIVE)
            verified.append(
                rule.model_copy(update={"verification": verification, "status": status})
            )

        updated = library.model_copy(update={"rules": verified})
        ctx.emit_json("rules_verified", updated)
        report = {
            "rules": len(verified),
            "active": accepted_count,
            "candidate": len(verified) - accepted_count,
            "episodes_targeted": len(targeted),
            "episodes_clean": len(clean),
            "held_out_limited": held_out_limited,
            "checks_failed": checks_failed,
        }
        ctx.emit_json("verify_rules_report", report)
        ctx.emit_text(
            "report",
            "# Verify rules\n\n"
            f"- rules: {len(verified)}\n"
            f"- active: {accepted_count}\n"
            f"- candidate: {len(verified) - accepted_count}\n"
            f"- gate: prevented >= {settings.min_prevented}, "
            f"collateral <= {settings.max_collateral}\n"
            f"- held-out limited: {held_out_limited}\n"
            f"- checks failed (no evidence): {checks_failed}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "rules": float(len(verified)),
                "active": float(accepted_count),
                "candidate": float(len(verified) - accepted_count),
            },
        )
