"""``compile_rules`` — turn observed friction into atomic runtime rules.

A rule is only useful if it can be *checked at runtime*, so each compiled rule
carries the TRACE structure: an applicability condition, an imperative
instruction, and a verifier the runtime can evaluate. Compilation is per episode
so the rule keeps its context; identical rules from different episodes are merged
deterministically and their evidence unioned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.core.json import dumps
from frictiontune.domain.episodes import Episode
from frictiontune.domain.ids import rule_id
from frictiontune.domain.rules import CompiledRules, RuleLibrary, RuntimeRule
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import (
    clip,
    gather_tolerant,
    read_events_by_episode,
    read_tool_schemas,
    render_transcript,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from frictiontune.core.json import JsonObject
    from frictiontune.domain.friction import FrictionEvent
    from frictiontune.pipeline.context import StageContext

_COMPILE_PROMPT = "rule_compiler"


class CompileRulesSettings(StageSettings):
    """Knobs for rule compilation."""

    batch_size: int = 32
    max_transcript_chars: int = 100_000
    max_rules_per_episode: int = 8
    # Bound on the tool-schema block. The schemas are shared across episodes and
    # joined verbatim; on a corpus with many tools that prompt can exceed a slot,
    # and the episode would be silently skipped.
    max_schema_chars: int = 8_000


def _render_friction(events: list[FrictionEvent]) -> str:
    lines: list[str] = []
    for event in events:
        turns = ", ".join(str(span.turn) for span in event.evidence)
        lines.append(
            f"{event.id} | {event.form}/{event.cause} | {event.leaf} | "
            f"{event.summary or event.form} | evidence turns: {turns}"
        )
    return "\n".join(lines)


class CompileRules(Stage):
    """Compile atomic, verifiable rules from friction events."""

    spec = StageSpec(
        name="compile_rules",
        tier=Tier.A,
        requires_llm=True,
        prompts=["rule_compiler"],
        roles=[RoleName.RULE_COMPILER],
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
        shown = events[: settings.max_rules_per_episode]
        result = await ctx.run.gateway.complete_structured(
            RoleName.RULE_COMPILER,
            role,
            _COMPILE_PROMPT,
            {
                "friction": _render_friction(shown),
                "context": clip(render_transcript(episode.messages), settings.max_transcript_chars),
                "tool_schemas": clip(
                    "\n".join(dumps(schema) for schema in schemas), settings.max_schema_chars
                )
                or "(none observed)",
                "max_rules": settings.max_rules_per_episode,
            },
            CompiledRules,
        )
        rules: list[RuntimeRule] = []
        for draft in result.rules[: settings.max_rules_per_episode]:
            identifier = rule_id(draft.instruction, draft.applicability)
            rules.append(
                RuntimeRule(
                    id=identifier,
                    # Attribute only to the events the compiler was actually shown;
                    # beyond that it never saw them, so claiming them would be false
                    # provenance in the exported rule artifact.
                    source_events=[event.id for event in shown],
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
        grouped = read_events_by_episode(ctx)
        schemas = read_tool_schemas(ctx)
        collected: list[RuntimeRule] = []
        episodes_seen = 0
        episodes_failed = 0
        batch: list[Episode] = []

        async def flush() -> None:
            nonlocal episodes_failed
            if not batch:
                return
            results = await gather_tolerant(
                lambda episode: self._compile_episode(
                    episode, grouped[episode.id], ctx, settings, schemas
                ),
                batch,
            )
            for result in results:
                if isinstance(result, BaseException):
                    # An episode whose rules cannot be compiled contributes no
                    # rules; it must not take the other episodes' rules with it.
                    episodes_failed += 1
                    continue
                collected.extend(result)
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
            "episodes_failed": episodes_failed,
            "rules_generated": len(collected),
            "rules_after_merge": len(rules),
        }
        ctx.emit_json("rules_report", report)
        ctx.emit_text(
            "report",
            "# Compile rules\n\n"
            f"- episodes compiled: {episodes_seen}\n"
            f"- episodes failed: {episodes_failed}\n"
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
