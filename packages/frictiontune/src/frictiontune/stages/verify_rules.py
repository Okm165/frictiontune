"""``verify_rules`` — keep the rule track safe with agentic replay.

Each rule is replayed against a held-out sample of historical episodes: would it
have fired, would it have prevented the targeted friction, and would it have
blocked correct behaviour (collateral)? Collateral is the failure mode that makes
rules dangerous, so the accept gate requires prevention and forbids collateral.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import FrictionEvent
from frictiontune.domain.responses import RuleVerdict
from frictiontune.domain.rules import RuleLibrary, RuleStatus, RuleVerification, RuntimeRule
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
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


def _read_events(ctx: StageContext) -> dict[str, str]:
    """Map episode id to a one-line friction summary for the verifier prompt."""
    summaries: dict[str, str] = defaultdict(str)
    for event in ctx.catalog.stream("friction_events", FrictionEvent):
        if not summaries[event.episode_id]:
            summaries[event.episode_id] = event.summary or f"{event.form}/{event.leaf}"
    return summaries


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
        lead=Lead.CORE,
        roi="High",
        justification="Keeps the rule track safe by measuring collateral before activation.",
        evidence=["agentic replay; collateral is the dominant rule risk"],
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
        summaries = _read_events(ctx)
        targeted, clean = _sample_pools(ctx, summaries, settings)

        verified: list[RuntimeRule] = []
        accepted_count = 0
        for rule in library.rules:
            episodes = (
                targeted[: settings.episodes_per_rule] + clean[: settings.clean_episodes_per_rule]
            )
            verdicts = await asyncio.gather(
                *(
                    self._check(
                        rule,
                        episode,
                        summaries.get(episode.id, ""),
                        ctx,
                        settings,
                    )
                    for episode in episodes
                )
            )
            fires = sum(1 for verdict in verdicts if verdict.fires)
            prevented = sum(1 for verdict in verdicts if verdict.fires and verdict.prevents)
            collateral = sum(1 for verdict in verdicts if verdict.collateral)
            mean_conf = (
                sum(verdict.confidence for verdict in verdicts) / len(verdicts) if verdicts else 0.0
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
        }
        ctx.emit_json("verify_rules_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Verify rules\n\n"
            f"- rules: {len(verified)}\n"
            f"- active: {accepted_count}\n"
            f"- candidate: {len(verified) - accepted_count}\n"
            f"- gate: prevented >= {settings.min_prevented}, "
            f"collateral <= {settings.max_collateral}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "rules": float(len(verified)),
                "active": float(accepted_count),
                "candidate": float(len(verified) - accepted_count),
            },
        )
