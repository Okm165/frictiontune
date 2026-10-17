"""``evaluate_replay`` — the finish line: replay base vs tuned from identical context.

Each held-out example is replayed through both the base model and the served
adapter from the *same* context, then scored by the shared judge. A dashboard of
independent signals is built with a seeded bootstrap interval, so a small or noisy
difference from the model change is visible as such rather than sold as a win.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import ConfigError, StageError
from frictiontune.core.json import json_block_validity
from frictiontune.domain.dataset import SFTRecord, TrainingExample
from frictiontune.domain.responses import ReplayVerdict
from frictiontune.domain.rules import RuleStatus, RuntimeRule
from frictiontune.llm.messages import ChatMessage
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.reporting.dashboard import Direction, build_dashboard, make_signal
from frictiontune.reporting.report import render_dashboard
from frictiontune.sft.renderer import to_chat_messages
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.pipeline.context import StageContext

_JUDGE_PROMPT = "replay_judge"

_SIGNALS: tuple[tuple[str, Direction], ...] = (
    ("friction_avoidance", "higher_better"),
    ("correctness", "higher_better"),
    ("clarity", "higher_better"),
    ("verbosity_chars", "lower_better"),
    ("format_valid", "higher_better"),
)


class EvaluateReplaySettings(StageSettings):
    """Knobs for the replay evaluation."""

    max_examples: int = 0  # 0 = all
    # The replay context budget in characters. 0 (the default in configs) means
    # *derive*: (max_seq_len - max_tokens - margin) x 3 — a conservative
    # chars-per-token for code-heavy text. Deriving is not a nicety: training
    # packs every record to exactly `max_seq_len` tokens, and GAP-103 measured
    # the replay feeding contexts 1.8-3x deeper than that — out-of-distribution
    # for the adapter (RoPE attention uncertainty grows beyond the trained
    # length), with a long prompt able to overflow the served window and
    # silently drop the system block. An explicitly set value still wins, for
    # experiments that want a different regime — and is then a declared
    # instrument change, not an accident.
    max_context_chars: int = 0
    max_generation_chars: int = 20_000
    sampling_temperature: float = 0.0
    max_tokens: int = 1_024
    # Inject the verified runtime rules into the replay context on BOTH sides
    # (GAP-095). The rules are the project's other deliverable; measuring the
    # adapter without them answers a question nobody is deploying. Off by
    # default so the weights-only comparison stays the baseline instrument.
    inject_rules: bool = False
    max_rules_chars: int = 12_000


def _load_active_rules(ctx: StageContext, settings: EvaluateReplaySettings) -> str:
    """Render the verified-active rules into a system block.

    Fails loudly when ``inject_rules`` is set but no rules exist: a rules-free
    replay silently passing as a rules-injected one would be a mislabeled
    measurement, which is worse than a failed phase.
    """
    if not ctx.catalog.has("rules_verified"):
        raise StageError(
            "evaluate_replay",
            "inject_rules is set but no rules_verified artifact exists; "
            "run verify_rules first or unset inject_rules",
        )
    payload = ctx.catalog.read_json("rules_verified")
    rules = payload.get("rules", []) if isinstance(payload, dict) else []
    active = [
        RuntimeRule.model_validate(rule)
        for rule in rules
        if isinstance(rule, dict) and rule.get("status") == RuleStatus.ACTIVE
    ]
    lines = [
        f"- when {rule.applicability}: {rule.instruction}" for rule in active if rule.instruction
    ]
    header = (
        "## Runtime rules\n\n"
        "Mined from friction observed in past sessions. Follow these while working:\n"
    )
    return clip(header + "\n".join(lines), settings.max_rules_chars)


def _messages_with_rules(
    messages: list[ChatMessage], rules_text: str, max_context_chars: int
) -> list[ChatMessage]:
    """Prepend the rules as a system message within the same context budget.

    The rules go in front *after* trimming, so they are never the turns the
    trimmer drops; the conversation instead gets the remaining budget, keeping
    the total sent to the served model bounded exactly as before.
    """
    budget = max_context_chars - len(rules_text)
    trimmed = _trim_context(messages, budget)
    return [ChatMessage(role="system", content=rules_text), *trimmed]


def _read_contamination(ctx: StageContext) -> tuple[int, int] | None:
    """Read ``(held_out, dropped)`` from the contamination report, if present.

    ``held_out`` is what the time split put on the evaluation side and
    ``dropped`` is how many of those the contamination tiers then removed — the
    two numbers that tell an empty holdout apart from a leaky one. Returns
    ``None`` when the report is unavailable, so the caller can say it does not
    know rather than assert a cause it has not checked.
    """
    if not ctx.catalog.has("contamination_report"):
        return None
    report = ctx.catalog.read_json("contamination_report")
    if not isinstance(report, dict):
        return None
    held_out = report.get("eval")
    dropped = report.get("dropped_contaminated")
    if isinstance(held_out, int) and isinstance(dropped, int):
        return held_out, dropped
    return None


def _context_budget(settings: EvaluateReplaySettings, max_seq_len: int) -> int:
    """Resolve the replay context budget in characters.

    0 means derive from the training window: the replay must measure the
    adapter *in distribution*, and training never saw a context deeper than
    `max_seq_len` tokens (GAP-103). The reserve subtracts the generation and a
    template margin; 3 chars/token is deliberately conservative for a code-heavy
    corpus, where tokenizers run worse than the prose-friendly 4.
    """
    if settings.max_context_chars > 0:
        return settings.max_context_chars
    reserve = settings.max_tokens + 256
    return max(1_000, (max_seq_len - reserve) * 3)


def _format_valid(text: str) -> float:
    """Structural validity of any JSON the continuation chose to emit.

    Empty output scores 0; otherwise it is the fraction of fenced JSON blocks
    that parse (1.0 when there are none). This measures the tool-call protocol,
    not meaning.
    """
    if not text.strip():
        return 0.0
    return json_block_validity(text)


def _trim_context(messages: Sequence[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """Keep the most recent messages within ``max_chars`` (the served model's window).

    The replay sends a held-out episode's context to the base and adapter servers;
    an episode longer than the served context would make every generation fail, so
    the tail (the part closest to the continuation) is kept and the oldest turns
    are dropped.
    """
    kept: list[ChatMessage] = []
    size = 0
    for message in reversed(messages):
        if kept and size + len(message.content) > max_chars:
            break
        kept.append(message)
        size += len(message.content)
    kept.reverse()
    return kept


def _replay_notes(
    settings: EvaluateReplaySettings,
    configured_seeds: Sequence[int],
    sampling_seeds: Sequence[int],
) -> list[str]:
    """Notes explaining how sampling was actually performed."""
    notes = ["lucky-pass flag and in-the-loop user simulation are evidence-gated extensions"]
    if settings.sampling_temperature <= 0.0 and len(configured_seeds) > 1:
        notes.append(
            f"temperature 0: collapsed {len(configured_seeds)} configured seeds to "
            "one deterministic sample per example"
        )
    elif len(sampling_seeds) > 1:
        notes.append(
            f"sampled {len(sampling_seeds)} seeded generations per example at "
            f"temperature {settings.sampling_temperature:g} (seeds {sampling_seeds})"
        )
    return notes


class EvaluateReplay(Stage):
    """Replay held-out episodes on base and adapter and build a dashboard."""

    spec = StageSpec(
        name="evaluate_replay",
        tier=Tier.A,
        requires_llm=True,
        uses_evaluation=True,
        prompts=["replay_judge"],
        roles=[RoleName.TARGET, RoleName.ADAPTER, RoleName.JUDGE],
        lead=Lead.CORE,
        roi="High",
        justification="The finish line: the friction drop is only credible from a clean replay.",
        evidence=["multi-signal dashboard; seeded bootstrap CI"],
        cost=CostProfile(compute="mixed", api_usd="high", maintenance="medium"),
        inputs={"eval": "jsonl", "judged": "jsonl", "adapter": "json"},
        # Read only to explain an empty holdout accurately; consumed when present.
        # rules_verified feeds the GAP-095 rules-in-replay arm when present.
        optional_inputs={"contamination_report": "json", "rules_verified": "json"},
        outputs={"dashboard": "json", "replay": "jsonl"},
        kill_criterion="Stop the project if the friction drop is within the interval.",
    )
    settings_model = EvaluateReplaySettings

    def _eval_ids(self, ctx: StageContext) -> list[str]:
        return [record.id for record in ctx.catalog.stream("eval", SFTRecord)]

    def _eval_examples(self, ctx: StageContext, ids: set[str]) -> list[TrainingExample]:
        return [
            example
            for example in ctx.catalog.stream("judged", TrainingExample)
            if example.id in ids and example.kept
        ]

    def _role(
        self, ctx: StageContext, role: RoleName, settings: EvaluateReplaySettings
    ) -> LlmRoleConfig:
        try:
            base = ctx.config.role(role)
        except ConfigError as exc:
            raise StageError(
                "evaluate_replay",
                f"role {role.value!r} is not configured; serve the adapter and add the role",
            ) from exc
        return base.model_copy(
            update={
                "temperature": settings.sampling_temperature,
                "max_tokens": settings.max_tokens,
                "cache": False,
            }
        )

    async def _generate(
        self,
        example: TrainingExample,
        role: RoleName,
        config: LlmRoleConfig,
        ctx: StageContext,
        settings: EvaluateReplaySettings,
        *,
        seed: int,
        budget: int,
        rules_text: str = "",
    ) -> str:
        messages = _trim_context(
            to_chat_messages(example.context, include_reasoning=False),
            budget,
        )
        if rules_text:
            messages = _messages_with_rules(messages, rules_text, budget)
        response = await ctx.run.gateway.complete_text(role, config, messages, seed=seed)
        return response.text

    async def _judge(
        self,
        example: TrainingExample,
        continuation: str,
        ctx: StageContext,
        settings: EvaluateReplaySettings,
    ) -> ReplayVerdict:
        try:
            role = ctx.config.role(RoleName.JUDGE)
        except ConfigError as exc:
            raise StageError("evaluate_replay", "the judge role is not configured") from exc
        # Pin the judge to greedy decoding: a sampled judge would add noise to
        # the very comparison it is meant to measure.
        role = role.model_copy(update={"temperature": 0.0, "cache": True})
        return await ctx.run.gateway.complete_structured(
            RoleName.JUDGE,
            role,
            _JUDGE_PROMPT,
            {
                "context": clip(render_transcript(example.context), settings.max_context_chars),
                "continuation": clip(continuation, settings.max_generation_chars),
            },
            ReplayVerdict,
        )

    async def _replay_one(
        self,
        example: TrainingExample,
        ctx: StageContext,
        settings: EvaluateReplaySettings,
        seeds: Sequence[int],
        rules_text: str = "",
        budget: int = 0,
    ) -> dict[str, dict[str, float]]:
        base_role = self._role(ctx, RoleName.TARGET, settings)
        adapter_role = self._role(ctx, RoleName.ADAPTER, settings)
        totals = {side: {name: 0.0 for name, _dir in _SIGNALS} for side in ("base", "adapter")}
        for seed in seeds:
            for side, role, config in (
                ("base", RoleName.TARGET, base_role),
                ("adapter", RoleName.ADAPTER, adapter_role),
            ):
                continuation = await self._generate(
                    example,
                    role,
                    config,
                    ctx,
                    settings,
                    seed=seed,
                    rules_text=rules_text,
                    budget=budget,
                )
                verdict = await self._judge(example, continuation, ctx, settings)
                totals[side]["friction_avoidance"] += verdict.friction_avoidance
                totals[side]["correctness"] += verdict.correctness
                totals[side]["clarity"] += verdict.clarity
                totals[side]["verbosity_chars"] += len(continuation)
                totals[side]["format_valid"] += _format_valid(continuation)
        count = max(1, len(seeds))
        return {
            side: {name: value / count for name, value in totals[side].items()}
            for side in ("base", "adapter")
        }

    def _skip_empty_holdout(self, ctx: StageContext, seeds: Sequence[int]) -> StageResult:
        """Report a skipped replay when the held-out set is empty.

        A strict time split can legitimately leave nothing to evaluate: on a
        single-user corpus the newest work usually continues the same task as the
        training data, so the contamination tiers can drop every held-out item as
        a leak and the eval number becomes vacuous (GAP-047). That is a property
        of the *corpus*, not a broken stage, and `export` only needs the adapter —
        so raising here would block a downstream stage for a reason the operator
        cannot act on at this point. The replay is therefore skipped with an
        explicit, well-formed empty dashboard and a note naming the cause.

        The note must not guess the cause. A split can also come out empty on its
        own — a large newest session straddling the boundary once sent every
        record to train — and blaming contamination then sends the reader to a
        report that says `dropped_contaminated: 0`.
        """
        contamination = _read_contamination(ctx)
        if contamination is None or contamination[0] == 0:
            cause = (
                "the time split put every curated record on the training side "
                "(see decontaminate's report and contamination_report)"
            )
        else:
            cause = (
                f"the time split held out {contamination[0]} record(s) and all of "
                f"them were then dropped as training leaks "
                f"(see decontaminate's contamination_report)"
            )
        note = f"held-out set is empty: {cause}. The adapter is exported without a replay number."
        dashboard = build_dashboard([], episodes=0, seeds=list(seeds), notes=[note])
        ctx.emit_json("dashboard", dashboard)
        ctx.emit_jsonl("replay", [])
        ctx.emit_text("report", render_dashboard(dashboard))
        return StageResult.from_context(
            ctx,
            metrics={"episodes": 0.0, "skipped": 1.0},
            notes=[note],
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Replay eval examples and emit the dashboard."""
        settings = ctx.settings(self.settings_model)
        seeds = ctx.config.evaluation.seeds or [0]
        # Extra seeds only buy samples when decoding is stochastic. At
        # temperature 0 every configured seed yields the identical generation,
        # which would fabricate confidence, so collapse to one deterministic
        # sample and say so.
        deterministic = settings.sampling_temperature <= 0.0
        sampling_seeds = [seeds[0]] if deterministic else list(seeds)
        ids = set(self._eval_ids(ctx))
        examples = self._eval_examples(ctx, ids)
        if settings.max_examples:
            examples = examples[: settings.max_examples]
        if not examples:
            return self._skip_empty_holdout(ctx, sampling_seeds)

        # Episodes within one session are not independent observations, so the
        # sign test's primary view aggregates to session means (GAP-097). The
        # eval records carry the session id; the judged examples match by id.
        session_of = {
            record.id: str(record.meta.get("session_id") or record.id)
            for record in ctx.catalog.stream("eval", SFTRecord)
        }
        cluster_ids = [session_of.get(example.id, example.id) for example in examples]

        # The GAP-095 arm: replay with the mined runtime rules in context, on
        # both sides, so the measured quantity is the deployed configuration —
        # (base+rules) vs (adapter+rules) — rather than weights alone.
        rules_text = ""
        if settings.inject_rules:
            rules_text = _load_active_rules(ctx, settings)
        budget = _context_budget(settings, ctx.config.hardware.max_seq_len)

        results = await asyncio.gather(
            *(
                self._replay_one(
                    example, ctx, settings, sampling_seeds, rules_text=rules_text, budget=budget
                )
                for example in examples
            )
        )

        baseline: dict[str, list[float]] = {name: [] for name, _dir in _SIGNALS}
        tuned: dict[str, list[float]] = {name: [] for name, _dir in _SIGNALS}
        for entry in results:
            for name, _dir in _SIGNALS:
                baseline[name].append(entry["base"][name])
                tuned[name].append(entry["adapter"][name])

        mcid = ctx.config.evaluation.min_important_delta
        signals = [
            make_signal(
                name,
                baseline[name],
                tuned[name],
                direction=direction,
                seed=ctx.config.seed,
                cluster_ids=cluster_ids,
                min_important_delta=mcid,
            )
            for name, direction in _SIGNALS
        ]
        rule_count = sum(1 for line in rules_text.splitlines() if line.startswith("- "))
        dashboard = build_dashboard(
            signals,
            episodes=len(examples),
            seeds=list(sampling_seeds),
            notes=_replay_notes(settings, seeds, sampling_seeds)
            + (
                [
                    f"rules injected into both sides: {rule_count} active rules, "
                    f"{len(rules_text)} chars"
                ]
                if rules_text
                else ["rules not injected (weights-only comparison)"]
            ),
            clusters=len(set(cluster_ids)),
            min_important_delta=mcid,
        )
        ctx.emit_json("dashboard", dashboard)
        ctx.emit_jsonl(
            "replay",
            [
                {"id": example.id, "seed_count": len(sampling_seeds), **entry}
                for example, entry in zip(examples, results, strict=True)
            ],
        )
        ctx.emit_text("report", render_dashboard(dashboard))
        return StageResult.from_context(
            ctx,
            metrics={
                "episodes": float(len(examples)),
                "friction_drop": dashboard.friction_drop,
                "friction_drop_ci_low": dashboard.friction_drop_ci_low,
                "friction_drop_ci_high": dashboard.friction_drop_ci_high,
            },
            notes=dashboard.notes,
        )
