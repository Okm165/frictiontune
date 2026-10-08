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
from frictiontune.core.errors import StageError
from frictiontune.core.json import json_block_validity
from frictiontune.domain.dataset import SFTRecord, TrainingExample
from frictiontune.domain.responses import JudgeVerdict
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.reporting.dashboard import Direction, build_dashboard, make_signal
from frictiontune.reporting.report import render_dashboard
from frictiontune.sft.renderer import to_chat_messages
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.pipeline.context import StageContext

_JUDGE_PROMPT = "judge"

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
    max_context_chars: int = 60_000
    max_generation_chars: int = 20_000
    sampling_temperature: float = 0.0
    max_tokens: int = 1_024


def _format_valid(text: str) -> float:
    """Structural validity of any JSON the continuation chose to emit.

    Empty output scores 0; otherwise it is the fraction of fenced JSON blocks
    that parse (1.0 when there are none). This measures the tool-call protocol,
    not meaning.
    """
    if not text.strip():
        return 0.0
    return json_block_validity(text)


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
            f"sampled {len(sampling_seeds)} generations per example at "
            f"temperature {settings.sampling_temperature:g}"
        )
    return notes


class EvaluateReplay(Stage):
    """Replay held-out episodes on base and adapter and build a dashboard."""

    spec = StageSpec(
        name="evaluate_replay",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="High",
        justification="The finish line: the friction drop is only credible from a clean replay.",
        evidence=["multi-signal dashboard; seeded bootstrap CI"],
        cost=CostProfile(compute="mixed", api_usd="high", maintenance="medium"),
        inputs={"eval": "jsonl", "judged": "jsonl", "adapter": "json"},
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
        except KeyError as exc:
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
    ) -> str:
        messages = to_chat_messages(example.context, include_reasoning=False)
        response = await ctx.run.gateway.complete_text(role, config, messages)
        return response.text

    async def _judge(
        self,
        example: TrainingExample,
        continuation: str,
        ctx: StageContext,
        settings: EvaluateReplaySettings,
    ) -> JudgeVerdict:
        try:
            role = ctx.config.role(RoleName.JUDGE)
        except KeyError as exc:
            raise StageError("evaluate_replay", "the judge role is not configured") from exc
        # Pin the judge to greedy decoding: a sampled judge would add noise to
        # the very comparison it is meant to measure.
        role = role.model_copy(update={"temperature": 0.0, "cache": True})
        return await ctx.run.gateway.complete_structured(
            RoleName.JUDGE,
            role,
            _JUDGE_PROMPT,
            {
                "candidate_id": f"{example.id}:replay",
                "source": example.source,
                "track": "replay",
                "context": clip(render_transcript(example.context), settings.max_context_chars),
                "completion": clip(continuation, settings.max_generation_chars),
            },
            JudgeVerdict,
        )

    async def _replay_one(
        self,
        example: TrainingExample,
        ctx: StageContext,
        settings: EvaluateReplaySettings,
        seeds: Sequence[int],
    ) -> dict[str, dict[str, float]]:
        base_role = self._role(ctx, RoleName.TARGET, settings)
        adapter_role = self._role(ctx, RoleName.ADAPTER, settings)
        totals = {side: {name: 0.0 for name, _dir in _SIGNALS} for side in ("base", "adapter")}
        for _seed in seeds:
            for side, role, config in (
                ("base", RoleName.TARGET, base_role),
                ("adapter", RoleName.ADAPTER, adapter_role),
            ):
                continuation = await self._generate(example, role, config, ctx)
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
            raise StageError("evaluate_replay", "no eval examples available")

        results = await asyncio.gather(
            *(self._replay_one(example, ctx, settings, sampling_seeds) for example in examples)
        )

        baseline: dict[str, list[float]] = {name: [] for name, _dir in _SIGNALS}
        tuned: dict[str, list[float]] = {name: [] for name, _dir in _SIGNALS}
        for entry in results:
            for name, _dir in _SIGNALS:
                baseline[name].append(entry["base"][name])
                tuned[name].append(entry["adapter"][name])

        signals = [
            make_signal(
                name, baseline[name], tuned[name], direction=direction, seed=ctx.config.seed
            )
            for name, direction in _SIGNALS
        ]
        dashboard = build_dashboard(
            signals,
            episodes=len(examples),
            seeds=list(sampling_seeds),
            lucky_pass_rate=0.0,
            notes=_replay_notes(settings, seeds, sampling_seeds),
        )
        ctx.emit_json("dashboard", dashboard)
        ctx.emit_jsonl(
            "replay",
            [
                {"id": example.id, "seed_count": len(sampling_seeds), **entry}
                for example, entry in zip(examples, results, strict=True)
            ],
        )
        ctx.emit_text("report.md", render_dashboard(dashboard))
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
