"""``compile_rules`` — turn observed friction into atomic runtime rules.

A rule is only useful if it can be *checked at runtime*, so each compiled rule
carries the TRACE structure: an applicability condition, an imperative
instruction, and a verifier the runtime can evaluate. Compilation is per episode
so the rule keeps its context; identical rules from different episodes are merged
deterministically and their evidence unioned.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import FrictionEvent
from frictiontune.domain.ids import rule_id
from frictiontune.domain.responses import InferredToolSchema
from frictiontune.domain.rules import CompiledRules, RuleLibrary, RuntimeRule
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from frictiontune.core.json import JsonObject
    from frictiontune.pipeline.context import StageContext

_COMPILE_PROMPT = "rule_compiler"


class CompileRulesSettings(StageSettings):
    """Knobs for rule compilation."""

    batch_size: int = 32
    max_transcript_chars: int = 100_000
    max_rules_per_episode: int = 8


def _render_friction(events: list[FrictionEvent]) -> str:
    lines: list[str] = []
    for event in events:
        turns = ", ".join(str(span.turn) for span in event.evidence)
        lines.append(
            f"{event.id} | {event.form}/{event.cause} | {event.leaf} | "
            f"{event.summary or event.form} | evidence turns: {turns}"
        )
    return "\n".join(lines)


def _read_events(ctx: StageContext) -> dict[str, list[FrictionEvent]]:
    grouped: dict[str, list[FrictionEvent]] = defaultdict(list)
    for event in ctx.catalog.stream("friction_events", FrictionEvent):
        grouped[event.episode_id].append(event)
    return grouped


def _read_schemas(ctx: StageContext) -> list[JsonObject]:
    if not ctx.catalog.has("tool_schemas"):
        return []
    return [
        schema.model_dump(mode="json")
        for schema in ctx.catalog.stream("tool_schemas", InferredToolSchema)
    ]


class CompileRules(Stage):
    """Compile atomic, verifiable rules from friction events."""

    spec = StageSpec(
        name="compile_rules",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="Very High",
        justification=(
            "Friction reduction with zero training, and a runtime artifact teams can adopt."
        ),
        evidence=["TRACE structure; artifact-grounded verifier preferred"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="medium"),
        inputs={"friction_events": "jsonl", "episodes": "jsonl", "tool_schemas": "jsonl"},
        outputs={"rules": "json", "rules_report": "json"},
        kill_criterion="Drop if artifact-grounded rules never pass the verifier gate.",
    )
    settings_model = CompileRulesSettings

    async def _compile_episode(
        self,
        episode: Episode,
        events: list[FrictionEvent],
        ctx: StageContext,
        settings: CompileRulesSettings,
        schemas: list[JsonObject],
    ) -> list[RuntimeRule]:
        role = ctx.config.role(RoleName.RULE_COMPILER)
        result = await ctx.run.gateway.complete_structured(
            RoleName.RULE_COMPILER,
            role,
            _COMPILE_PROMPT,
            {
                "friction": _render_friction(events[: settings.max_rules_per_episode]),
                "context": clip(render_transcript(episode.messages), settings.max_transcript_chars),
                "tool_schemas": "\n".join(str(schema) for schema in schemas) or "(none observed)",
            },
            CompiledRules,
        )
        rules: list[RuntimeRule] = []
        for draft in result.rules[: settings.max_rules_per_episode]:
            identifier = rule_id(draft.instruction, draft.applicability)
            rules.append(
                RuntimeRule(
                    id=identifier,
                    source_events=[event.id for event in events],
                    applicability=draft.applicability,
                    instruction=draft.instruction,
                    scope=draft.scope,
                    verifier=draft.verifier,
                    rationale=draft.rationale,
                )
            )
        return rules

    def _merge(self, rules: Iterable[RuntimeRule]) -> list[RuntimeRule]:
        """Merge rules with identical ids, unioning their source events."""
        merged: dict[str, RuntimeRule] = {}
        for rule in rules:
            existing = merged.get(rule.id)
            if existing is None:
                merged[rule.id] = rule
                continue
            merged[rule.id] = existing.model_copy(
                update={
                    "source_events": sorted(set(existing.source_events) | set(rule.source_events))
                }
            )
        return list(merged.values())

    def _episodes_with_events(
        self, ctx: StageContext, grouped: dict[str, list[FrictionEvent]]
    ) -> Iterator[Episode]:
        for episode in ctx.catalog.stream("episodes", Episode):
            if episode.id in grouped:
                yield episode

    async def run(self, ctx: StageContext) -> StageResult:
        """Compile and merge rules across every friction episode."""
        settings = ctx.settings(self.settings_model)
        grouped = _read_events(ctx)
        schemas = _read_schemas(ctx)
        collected: list[RuntimeRule] = []
        episodes_seen = 0
        batch: list[Episode] = []

        async def flush() -> None:
            if not batch:
                return
            results = await asyncio.gather(
                *(
                    self._compile_episode(episode, grouped[episode.id], ctx, settings, schemas)
                    for episode in batch
                )
            )
            for rules in results:
                collected.extend(rules)
            batch.clear()

        for episode in self._episodes_with_events(ctx, grouped):
            episodes_seen += 1
            batch.append(episode)
            if len(batch) >= settings.batch_size:
                await flush()
        await flush()

        rules = self._merge(collected)
        library = RuleLibrary(rules=rules, generated_from="compile_rules")
        ctx.emit_json("rules", library)
        report = {
            "episodes_compiled": episodes_seen,
            "rules_generated": len(collected),
            "rules_after_merge": len(rules),
        }
        ctx.emit_json("rules_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Compile rules\n\n"
            f"- episodes compiled: {episodes_seen}\n"
            f"- rules generated: {len(collected)}\n"
            f"- rules after merge: {len(rules)}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "episodes_compiled": float(episodes_seen),
                "rules_generated": float(len(collected)),
                "rules": float(len(rules)),
            },
        )
