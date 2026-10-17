"""Evidence-gated extension stages (Tier C).

These are the demoted steps of the plan. Each is real and runnable, but only
after the core pipeline has produced evidence that justifies its cost; the
orchestrator refuses to run a Tier C stage unless ``allow_gated`` is set.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from frictiontune.backends.models import (
    GrpoRequest,
    RolloutRequest,
    RolloutResult,
    TrainingRequest,
    TrainingResult,
)
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import ConfigError, LlmError, StageError
from frictiontune.core.hashing import file_sha256
from frictiontune.domain.base import BaseProbe
from frictiontune.domain.dataset import PreferencePair, SFTRecord, TrainingExample
from frictiontune.domain.episodes import Episode, ReducedSpan
from frictiontune.domain.friction import FrictionEvent
from frictiontune.domain.ids import example_id
from frictiontune.domain.profiles import (
    Attribution,
    BehaviorDimension,
    EventAttribution,
    PreferenceProfile,
)
from frictiontune.domain.responses import (
    FrictionClusters,
    ProbeVerdict,
    ReduceSummary,
    SimulatedUserTurn,
    SyntheticBatch,
)
from frictiontune.domain.trace import Message, TextPart
from frictiontune.llm.messages import ChatMessage
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.reporting.dashboard import (
    Direction,
    build_dashboard,
    make_signal,
    relative_drop,
)
from frictiontune.reporting.report import render_dashboard
from frictiontune.stages.common import clip, read_events_by_episode, render_transcript
from frictiontune.stages.evaluate_replay import _format_valid
from frictiontune.stages.train_sft import TrainSftSettings, _epochs_for

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.core.json import JsonObject
    from frictiontune.pipeline.context import RunContext, StageContext

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EXT = Lead.EXTENSION


def _gated(reason: str) -> str:
    return reason


def _message(role: str, text: str, *, session_id: str, seq: int) -> Message:
    """Build a synthetic single-part message for generated examples."""
    return Message(
        id=f"{session_id}#{seq}",
        session_id=session_id,
        seq=seq,
        role=role,
        time_created=0,
        parts=[TextPart(text=text)],
    )


def _chat_transcript(messages: Sequence[ChatMessage], limit: int = 60_000) -> str:
    lines = [f"{message.role}: {message.content}" for message in messages]
    return clip("\n".join(lines), limit)


def _covered_events(clusters: Sequence[BehaviorDimension], known: set[str]) -> int:
    """Count the distinct known events named by any behaviour cluster.

    An event may be claimed by more than one cluster; coverage is a fraction of
    the events actually covered, so each event counts once.
    """
    return len(
        {event_id for cluster in clusters for event_id in cluster.event_ids if event_id in known}
    )


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


class ReduceSettings(StageSettings):
    """Knobs for episode-span reduction."""

    min_chars: int = 40_000
    chunk_turns: int = Field(default=40, ge=1)
    max_transcript_chars: int = 60_000


class Reduce(Stage):
    """Map-reduce compression of long episodes before analysis."""

    spec = StageSpec(
        name="reduce",
        tier=Tier.C,
        requires_llm=True,
        prompts=["reduce"],
        lead=_EXT,
        roi="Medium",
        justification=(
            "Long episodes exceed the analyst window; a faithful reduction keeps them analysable."
        ),
        evidence=["tool output p90 13 KB; episodes up to 102 assistant turns"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="medium"),
        inputs={"episodes": "jsonl"},
        outputs={"reduced_spans": "jsonl", "reduce_report": "json"},
        kill_criterion="Drop if compaction in normalize already fits every episode in one window.",
        gated_by=_gated("only needed when episodes exceed the analysis context window"),
    )
    settings_model = ReduceSettings

    async def _reduce_chunk(
        self,
        episode: Episode,
        start: int,
        ctx: StageContext,
        settings: ReduceSettings,
    ) -> ReducedSpan:
        chunk = episode.messages[start : start + settings.chunk_turns]
        end = start + len(chunk) - 1
        role = ctx.config.role(RoleName.COMPACTOR)
        result = await ctx.run.gateway.complete_structured(
            RoleName.COMPACTOR,
            role,
            "reduce",
            {
                "episode_id": episode.id,
                "start_turn": start,
                "end_turn": end,
                "char_count": sum(len(message.content_text()) for message in chunk),
                "transcript": clip(render_transcript(chunk), settings.max_transcript_chars),
            },
            ReduceSummary,
        )
        return ReducedSpan(
            episode_id=episode.id,
            start_turn=start,
            end_turn=end,
            summary=result.summary,
            preserved_failures=result.preserved_failures,
            dropped_turns=result.dropped_turns,
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Emit reduced spans for every long episode."""
        settings = ctx.settings(self.settings_model)
        sink = ctx.jsonl_sink("reduced_spans")
        reduced = 0
        episodes_seen = 0

        for episode in ctx.catalog.stream("episodes", Episode):
            if episode.features.char_count < settings.min_chars:
                continue
            episodes_seen += 1
            chunks = range(0, len(episode.messages), settings.chunk_turns)
            spans = await asyncio.gather(
                *(self._reduce_chunk(episode, start, ctx, settings) for start in chunks)
            )
            for span in spans:
                sink.write(span)
                reduced += 1
        sink.close()

        ctx.emit_json("reduce_report", {"episodes": episodes_seen, "spans": reduced})
        return StageResult.from_context(
            ctx,
            metrics={"episodes": float(episodes_seen), "spans": float(reduced)},
        )


# ---------------------------------------------------------------------------
# attribute
# ---------------------------------------------------------------------------


class AttributeSettings(StageSettings):
    """Knobs for root-cause attribution."""

    max_transcript_chars: int = 100_000


class Attribute(Stage):
    """Localise the earliest decisive step for each friction event."""

    spec = StageSpec(
        name="attribute",
        tier=Tier.C,
        requires_llm=True,
        prompts=["attribution"],
        lead=_EXT,
        roi="Medium",
        justification=(
            "Attribution sharpens masking and rule scoping, but rules and"
            " recoveries already work without it."
        ),
        evidence=["everest earliest decisive step; PRM literature"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="low"),
        inputs={"friction_events": "jsonl", "episodes": "jsonl"},
        outputs={"attributions": "jsonl", "attribute_report": "json"},
        kill_criterion="Drop if decisive turns never change masking or rule scope in practice.",
        gated_by=_gated("masking already works from recovery spans"),
    )
    settings_model = AttributeSettings

    async def _attribute(
        self,
        episode: Episode,
        event: FrictionEvent,
        ctx: StageContext,
        settings: AttributeSettings,
    ) -> EventAttribution:
        role = ctx.config.role(RoleName.ATTRIBUTOR)
        evidence_turns = ", ".join(str(span.turn) for span in event.evidence)
        result: Attribution = await ctx.run.gateway.complete_structured(
            RoleName.ATTRIBUTOR,
            role,
            "attribution",
            {
                "friction_summary": event.summary or f"{event.form}/{event.leaf}",
                "evidence_turns": evidence_turns,
                "causes": ", ".join(ctx.config.taxonomy.causes),
                "transcript": clip(
                    render_transcript(episode.messages), settings.max_transcript_chars
                ),
            },
            Attribution,
        )
        return EventAttribution(event_id=event.id, attribution=result)

    async def run(self, ctx: StageContext) -> StageResult:
        """Attribute every friction event that has an episode."""
        settings = ctx.settings(self.settings_model)
        grouped = read_events_by_episode(ctx)
        sink = ctx.jsonl_sink("attributions")
        done = 0
        for episode in ctx.catalog.stream("episodes", Episode):
            events = grouped.get(episode.id)
            if not events:
                continue
            results = await asyncio.gather(
                *(self._attribute(episode, event, ctx, settings) for event in events)
            )
            for result in results:
                sink.write(result)
                done += 1
        sink.close()
        ctx.emit_json("attribute_report", {"attributed": done})
        return StageResult.from_context(ctx, metrics={"attributed": float(done)})


# ---------------------------------------------------------------------------
# friction_cluster
# ---------------------------------------------------------------------------


class FrictionClusterSettings(StageSettings):
    """Knobs for behaviour clustering."""

    max_events: int = 400
    max_events_chars: int = 12_000


class FrictionCluster(Stage):
    """Group friction events into behaviour clusters (the Preference Profile)."""

    spec = StageSpec(
        name="friction_cluster",
        tier=Tier.C,
        requires_llm=True,
        prompts=["friction_cluster"],
        lead=_EXT,
        roi="Medium",
        justification=(
            "Turns a long tail of events into a handful of target behaviours"
            " for synthesis and rules."
        ),
        evidence=["Behaviour-cluster preference profiles"],
        cost=CostProfile(compute="llm", api_usd="low", maintenance="low"),
        inputs={"friction_events": "jsonl"},
        outputs={"preference_profile": "json", "friction_cluster_report": "json"},
        kill_criterion=(
            "Drop if clusters add nothing over the taxonomy counts in the friction profile."
        ),
        gated_by=_gated("only useful once the event tail is long"),
    )
    settings_model = FrictionClusterSettings

    def _render(self, events: Sequence[FrictionEvent]) -> str:
        return "\n".join(
            f"{event.id} | {event.form}/{event.cause} | {event.leaf} | "
            f"{event.summary or event.form}"
            for event in events
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Cluster events and emit the preference profile."""
        settings = ctx.settings(self.settings_model)
        events = list(ctx.catalog.stream("friction_events", FrictionEvent))[: settings.max_events]
        if not events:
            profile = PreferenceProfile()
        else:
            role = ctx.config.role(RoleName.FRICTION_ANALYST)
            clusters: FrictionClusters = await ctx.run.gateway.complete_structured(
                RoleName.FRICTION_ANALYST,
                role,
                "friction_cluster",
                {"events": clip(self._render(events), settings.max_events_chars)},
                FrictionClusters,
            )
            known = {event.id for event in events}
            covered = _covered_events(clusters.clusters, known)
            profile = PreferenceProfile(
                dimensions=clusters.clusters,
                total_events=len(events),
                covered_events=covered,
            )
        ctx.emit_json("preference_profile", profile)
        ctx.emit_json(
            "friction_cluster_report",
            {
                "events": len(events),
                "dimensions": len(profile.dimensions),
                "coverage": profile.coverage(),
            },
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "events": float(len(events)),
                "dimensions": float(len(profile.dimensions)),
                "coverage": profile.coverage(),
            },
        )


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class SynthesizeSettings(StageSettings):
    """Knobs for synthetic example generation."""

    per_dimension: int = 8
    max_dimensions: int = 10
    seeds_per_dimension: int = 5
    max_seeds_chars: int = 12_000


class Synthesize(Stage):
    """Generate synthetic friction-avoiding examples per behaviour dimension."""

    spec = StageSpec(
        name="synthesize",
        tier=Tier.C,
        requires_llm=True,
        prompts=["synthesize"],
        lead=_EXT,
        roi="Medium",
        justification=(
            "Rare behaviours need generated coverage; synthesis is optional"
            " while real recoveries remain."
        ),
        evidence=["RAFT/SRFT synthesis; PRIME"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="medium"),
        inputs={"preference_profile": "json", "candidates": "jsonl"},
        outputs={"synthetic": "jsonl", "synthesize_report": "json"},
        kill_criterion=(
            "Drop if synthetic examples fail the judge at a higher rate than mined ones."
        ),
        gated_by=_gated("mined recoveries may already cover the needed behaviours"),
    )
    settings_model = SynthesizeSettings

    def _seed_index(self, ctx: StageContext) -> dict[str, str]:
        seeds: dict[str, str] = {}
        for example in ctx.catalog.stream("candidates", TrainingExample):
            event_id = str(example.meta.get("event_id", ""))
            if event_id and event_id not in seeds:
                seeds[event_id] = render_transcript(example.completion)
        return seeds

    async def run(self, ctx: StageContext) -> StageResult:
        """Synthesize examples for each behaviour dimension."""
        settings = ctx.settings(self.settings_model)
        profile = PreferenceProfile.model_validate(ctx.catalog.read_json("preference_profile"))
        seeds = self._seed_index(ctx)
        role = ctx.config.role(RoleName.TEACHER)
        sink = ctx.jsonl_sink("synthetic")
        produced = 0

        for dimension in profile.dimensions[: settings.max_dimensions]:
            seed_texts = [seeds[event_id] for event_id in dimension.event_ids if event_id in seeds][
                : settings.seeds_per_dimension
            ]
            batch: SyntheticBatch = await ctx.run.gateway.complete_structured(
                RoleName.TEACHER,
                role,
                "synthesize",
                {
                    "dimension": dimension.dimension or dimension.name,
                    "target_behavior": dimension.target_behavior,
                    "count": settings.per_dimension,
                    "seeds": clip("\n\n---\n\n".join(seed_texts), settings.max_seeds_chars)
                    or "(no seeds available)",
                },
                SyntheticBatch,
            )
            for index, example in enumerate(batch.examples):
                identifier = example_id(dimension.name, "synthetic", produced + index)
                sink.write(
                    TrainingExample(
                        id=identifier,
                        source=dimension.name,
                        track="synthetic",
                        context=[_message("user", example.context, session_id=identifier, seq=0)],
                        completion=[
                            _message("assistant", example.completion, session_id=identifier, seq=1)
                        ],
                        meta={
                            "dimension": dimension.name,
                            "target_behavior": dimension.target_behavior,
                            "rationale": example.rationale,
                            "generated": True,
                        },
                    )
                )
            produced += len(batch.examples)
        sink.close()

        ctx.emit_json(
            "synthesize_report",
            {"dimensions": len(profile.dimensions), "synthetic": produced},
        )
        return StageResult.from_context(
            ctx,
            metrics={"dimensions": float(len(profile.dimensions)), "synthetic": float(produced)},
        )


# ---------------------------------------------------------------------------
# raft_srft
# ---------------------------------------------------------------------------


class RaftSrftSettings(StageSettings):
    """Knobs for building RAFT/SRFT success examples."""

    max_episodes: int = 500
    min_assistant_turns: int = 2


class RaftSrft(Stage):
    """Build success-track examples from clean episodes (RAFT/SRFT)."""

    spec = StageSpec(
        name="raft_srft",
        tier=Tier.C,
        lead=_EXT,
        roi="Medium",
        justification=(
            "Successful trajectories add a positive track, but recovery data is the core signal."
        ),
        evidence=["RAFT 2504.11343; SRFT 2605.10674"],
        cost=CostProfile(compute="cpu", api_usd="none", maintenance="low"),
        inputs={"episodes": "jsonl", "friction_events": "jsonl"},
        outputs={"sft_raft": "jsonl", "raft_report": "json"},
        kill_criterion="Drop if success-track data shifts the model away from recovery behaviour.",
        gated_by=_gated("recovery data may be sufficient"),
    )
    settings_model = RaftSrftSettings

    def _clean_episode_example(
        self, episode: Episode, index: int, *, min_assistant_turns: int
    ) -> TrainingExample | None:
        assistant_positions = [
            position
            for position, message in enumerate(episode.messages)
            if message.role == "assistant"
        ]
        if len(assistant_positions) < min_assistant_turns:
            return None
        first = assistant_positions[0]
        context = episode.messages[:first]
        completion = episode.messages[first:]
        if not context or not completion:
            return None
        return TrainingExample(
            id=example_id(episode.id, "success", index),
            source=episode.id,
            track="success",
            context=list(context),
            completion=list(completion),
            meta={"time_created": episode.time_created, "track": "success"},
        )

    async def run(self, ctx: StageContext) -> StageResult:
        """Emit success examples from episodes without friction events."""
        settings = ctx.settings(self.settings_model)
        grouped = read_events_by_episode(ctx)
        sink = ctx.jsonl_sink("sft_raft")
        produced = 0
        seen = 0
        for episode in ctx.catalog.stream("episodes", Episode):
            if episode.id in grouped:
                continue
            if seen >= settings.max_episodes:
                break
            seen += 1
            example = self._clean_episode_example(
                episode, produced, min_assistant_turns=settings.min_assistant_turns
            )
            if example is None:
                continue
            sink.write(example)
            produced += 1
        sink.close()
        ctx.emit_json("raft_report", {"episodes": seen, "examples": produced})
        return StageResult.from_context(
            ctx, metrics={"episodes": float(seen), "examples": float(produced)}
        )


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


class RolloutSettings(StageSettings):
    """Knobs for sandbox rollouts."""

    tasks: list[str] = Field(default_factory=list)
    max_turns: int = 30
    sandbox: str = "local"
    max_tasks: int = 20


class Rollout(Stage):
    """Run the tuned model against tasks in a sandbox."""

    spec = StageSpec(
        name="rollout",
        tier=Tier.C,
        lead=_EXT,
        roi="Medium",
        justification=(
            "Executable rollouts are the strongest evidence, but the sandbox"
            " is expensive to maintain."
        ),
        evidence=["sandbox rollout; open-ended SWE tasks"],
        cost=CostProfile(
            compute="mixed", api_usd="medium", human="calibration-slice", maintenance="high"
        ),
        inputs={"adapter": "json"},
        outputs={"rollouts": "jsonl", "rollout_report": "json"},
        kill_criterion=(
            "Drop the sandbox if judge-scored replay already predicts friction reduction."
        ),
        gated_by=_gated("requires a maintained execution sandbox"),
    )
    settings_model = RolloutSettings

    async def run(self, ctx: StageContext) -> StageResult:
        """Roll out the adapter on each configured task."""
        settings = ctx.settings(self.settings_model)
        if not settings.tasks:
            raise StageError("rollout", "no tasks configured under stages.rollout.tasks")
        adapter = ctx.catalog.read_json("adapter")
        adapter_dir = str(adapter.get("adapter_dir", "")) if isinstance(adapter, dict) else ""
        backend = ctx.run.backends.rollout()
        sink = ctx.jsonl_sink("rollouts")
        finished = 0
        for task in settings.tasks[: settings.max_tasks]:
            result: RolloutResult = await backend.rollout(
                RolloutRequest(
                    model=adapter_dir or "adapter",
                    adapter_dir=adapter_dir or None,
                    task_prompt=task,
                    max_turns=settings.max_turns,
                    sandbox=settings.sandbox,
                    seed=ctx.config.seed,
                )
            )
            finished += int(result.finished)
            sink.write({"task": task, "result": result.model_dump(mode="json")})
        sink.close()
        ctx.emit_json(
            "rollout_report",
            {"tasks": len(settings.tasks[: settings.max_tasks]), "finished": finished},
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "tasks": float(len(settings.tasks[: settings.max_tasks])),
                "finished": float(finished),
            },
        )


# ---------------------------------------------------------------------------
# train_grpo
# ---------------------------------------------------------------------------


class TrainGrpoSettings(StageSettings):
    """Knobs for the RL stage."""

    group_size: int = 8
    max_steps: int = -1
    learning_rate: float = 1e-5
    max_prompts: int = 2_000


class TrainGrpo(Stage):
    """Run GRPO with a composed reward (judge + format + tool validity)."""

    spec = StageSpec(
        name="train_grpo",
        tier=Tier.C,
        lead=_EXT,
        roi="Low",
        justification=(
            "RL can push past SFT, but only after SFT+RAFT has plateaued and a"
            " reward is trustworthy."
        ),
        evidence=["GRPO; PRIME reward modelling; anti-hacking"],
        cost=CostProfile(
            compute="gpu", api_usd="medium", human="calibration-slice", maintenance="high"
        ),
        inputs={"judged": "jsonl", "adapter": "json"},
        outputs={"grpo_result": "json", "grpo_prompts": "jsonl", "train_grpo_report": "json"},
        kill_criterion="Drop if reward hacking appears or SFT already saturates the signals.",
        gated_by=_gated("SFT must plateau first, and the reward must be calibrated"),
    )
    settings_model = TrainGrpoSettings

    def _prompts(self, ctx: StageContext) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for example in ctx.catalog.stream("judged", TrainingExample):
            if not example.kept:
                continue
            rows.append({"prompt": render_transcript(example.context)})
        return rows

    async def run(self, ctx: StageContext) -> StageResult:
        """Build prompts and delegate the GRPO job to the training backend."""
        settings = ctx.settings(self.settings_model)
        adapter = ctx.catalog.read_json("adapter")
        adapter_dir = str(adapter.get("adapter_dir", "")) if isinstance(adapter, dict) else ""
        base_model = str(adapter.get("base_model", "")) if isinstance(adapter, dict) else ""
        if not base_model:
            raise StageError("train_grpo", "adapter artifact is missing base_model")

        prompts = self._prompts(ctx)[: settings.max_prompts]
        prompts_ref = ctx.emit_jsonl("grpo_prompts", prompts)
        request = GrpoRequest(
            base_model=base_model,
            adapter_dir=adapter_dir or None,
            prompts_file=prompts_ref.path,
            output_dir=str(ctx.store.directory / "grpo"),
            hardware=ctx.config.hardware,
            seed=ctx.config.seed,
            learning_rate=settings.learning_rate,
            max_steps=settings.max_steps,
            group_size=settings.group_size,
        )
        result: TrainingResult = await ctx.run.backends.training().train_grpo(request)
        ctx.emit_json("grpo_result", result)
        ctx.emit_json(
            "train_grpo_report",
            {
                "prompts": len(prompts),
                "steps": result.steps,
                "reward": result.metrics.get("reward", 0.0),
            },
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "prompts": float(len(prompts)),
                "steps": float(result.steps),
                "reward": result.metrics.get("reward", 0.0),
            },
        )


# ---------------------------------------------------------------------------
# evaluate_sim_user
# ---------------------------------------------------------------------------


class EvaluateSimUserSettings(StageSettings):
    """Knobs for simulated-user-in-the-loop evaluation."""

    goals: list[str] = Field(default_factory=list)
    persona: str = "a pragmatic senior engineer who wants the task done correctly"
    max_turns: int = 12
    max_generation_chars: int = 20_000


class EvaluateSimUser(Stage):
    """Drive a simulated user against base and tuned models, counting user turns."""

    spec = StageSpec(
        name="evaluate_sim_user",
        tier=Tier.C,
        requires_llm=True,
        prompts=["user_simulator"],
        lead=_EXT,
        roi="Medium",
        justification=(
            "The user-in-the-loop signal is the most faithful, but it is"
            " costly and noisy at small scale."
        ),
        evidence=["AgentLens lucky passes; simulated user"],
        cost=CostProfile(compute="llm", api_usd="high", maintenance="medium"),
        inputs={"adapter": "json"},
        outputs={"sim_user": "json", "evaluate_sim_user_report": "json"},
        kill_criterion="Drop if judge-scored replay already ranks the two models the same way.",
        gated_by=_gated("replay dashboard is cheaper and already informative"),
    )
    settings_model = EvaluateSimUserSettings

    def _role(self, ctx: StageContext, role: RoleName, temperature: float) -> LlmRoleConfig:
        try:
            base = ctx.config.role(role)
        except ConfigError as exc:
            raise StageError("evaluate_sim_user", f"role {role.value!r} is not configured") from exc
        return base.model_copy(update={"temperature": temperature, "cache": False})

    async def _run_scenario(
        self,
        goal: str,
        role: RoleName,
        config: LlmRoleConfig,
        ctx: StageContext,
        settings: EvaluateSimUserSettings,
    ) -> int:
        sim_role = ctx.config.role(RoleName.USER_SIMULATOR)
        history: list[ChatMessage] = []
        # Count the simulated user's turns, not the model's replies: the signal is
        # how much user steering a goal needs, so the assistant response on the
        # same exchange must not inflate it.
        turns = 0
        for _ in range(settings.max_turns):
            turn: SimulatedUserTurn = await ctx.run.gateway.complete_structured(
                RoleName.USER_SIMULATOR,
                sim_role,
                "user_simulator",
                {
                    "goal": goal,
                    "persona": settings.persona,
                    "transcript": _chat_transcript(history),
                },
                SimulatedUserTurn,
            )
            history.append(ChatMessage.user(turn.message))
            turns += 1
            if turn.resolved:
                break
            response = await ctx.run.gateway.complete_text(role, config, history)
            history.append(
                ChatMessage.assistant(clip(response.text, settings.max_generation_chars))
            )
        return turns

    async def run(self, ctx: StageContext) -> StageResult:
        """Run each goal against both models and build a turn-count dashboard."""
        settings = ctx.settings(self.settings_model)
        if not settings.goals:
            raise StageError(
                "evaluate_sim_user", "no goals configured under stages.evaluate_sim_user.goals"
            )
        base_role = self._role(ctx, RoleName.TARGET, 0.0)
        adapter_role = self._role(ctx, RoleName.ADAPTER, 0.0)

        base_turns: list[float] = []
        adapter_turns: list[float] = []
        for goal in settings.goals:
            base, adapter = await asyncio.gather(
                self._run_scenario(goal, RoleName.TARGET, base_role, ctx, settings),
                self._run_scenario(goal, RoleName.ADAPTER, adapter_role, ctx, settings),
            )
            base_turns.append(float(base))
            adapter_turns.append(float(adapter))

        signal = make_signal(
            "user_turns_per_episode",
            base_turns,
            adapter_turns,
            direction="lower_better",
            seed=ctx.config.seed,
        )
        dashboard = build_dashboard(
            [signal],
            episodes=len(settings.goals),
            seeds=[ctx.config.seed],
            notes=["simulated user; small samples are noisy"],
        )
        ctx.emit_json("sim_user", dashboard)
        ctx.emit_json(
            "evaluate_sim_user_report",
            {
                "goals": len(settings.goals),
                "turn_drop": relative_drop(signal.baseline, signal.tuned),
            },
        )
        ctx.emit_text("report", render_dashboard(dashboard))
        return StageResult.from_context(
            ctx,
            metrics={
                "goals": float(len(settings.goals)),
                "turn_drop": relative_drop(signal.baseline, signal.tuned),
            },
        )


# ---------------------------------------------------------------------------
# evaluate_external
# ---------------------------------------------------------------------------


class EvaluateExternalSettings(StageSettings):
    """Knobs for external benchmark calibration."""

    max_tasks: int = 200
    max_context_chars: int = 60_000
    max_tokens: int = 1_024


class EvaluateExternal(Stage):
    """Calibrate the friction signals against external benchmark corpora."""

    spec = StageSpec(
        name="evaluate_external",
        tier=Tier.C,
        requires_llm=True,
        uses_evaluation=True,
        prompts=["judge"],
        lead=_EXT,
        roi="Medium",
        justification=(
            "External calibration guards against overfitting to one corpus,"
            " but only matters once the core works."
        ),
        evidence=["active-learning calibration; external benchmarks"],
        cost=CostProfile(compute="mixed", api_usd="high", maintenance="medium"),
        inputs={"adapter": "json"},
        outputs={"external": "json", "evaluate_external_report": "json"},
        kill_criterion="Drop if the internal replay and sim-user signals already generalise.",
        gated_by=_gated("internal replay is the primary evaluation"),
    )
    settings_model = EvaluateExternalSettings

    def external_input_hashes(self, run: RunContext) -> list[str]:
        """Hash each configured external corpus so a changed corpus is not cached.

        The corpora are read from disk, not the artifact catalog, so without this
        the run id would not change when a corpus file changed and a stale
        external evaluation would be reused.
        """
        return [
            file_sha256(path)
            for path in (Path(item) for item in run.config.evaluation.external_corpora)
            if path.is_file()
        ]

    def _load_tasks(self, ctx: StageContext) -> list[dict[str, str]]:
        from frictiontune.core.json import iter_jsonl

        tasks: list[dict[str, str]] = []
        for path_str in ctx.config.evaluation.external_corpora:
            path = Path(path_str)
            if not path.is_file():
                ctx.logger.warning("external corpus missing", path=path_str)
                continue
            for row in iter_jsonl(path, skip_invalid=True):
                if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                    tasks.append({"prompt": str(row["prompt"])})
        return tasks

    async def _judge_external(
        self,
        prompt: str,
        completion: str,
        ctx: StageContext,
        settings: EvaluateExternalSettings,
    ) -> float:
        from frictiontune.domain.responses import JudgeVerdict

        role = ctx.config.role(RoleName.JUDGE)
        verdict: JudgeVerdict = await ctx.run.gateway.complete_structured(
            RoleName.JUDGE,
            role,
            "judge",
            {
                "candidate_id": "external",
                "source": "external",
                "track": "success",
                "context": clip(prompt, settings.max_context_chars),
                "completion": clip(completion, settings.max_context_chars),
            },
            JudgeVerdict,
        )
        return float(verdict.total())

    async def run(self, ctx: StageContext) -> StageResult:
        """Generate and judge external tasks for both models."""
        settings = ctx.settings(self.settings_model)
        tasks = self._load_tasks(ctx)[: settings.max_tasks]
        if not tasks:
            raise StageError("evaluate_external", "no external corpora configured or readable")

        base_role = ctx.config.role(RoleName.TARGET).model_copy(
            update={"max_tokens": settings.max_tokens}
        )
        adapter_role = ctx.config.role(RoleName.ADAPTER).model_copy(
            update={"max_tokens": settings.max_tokens}
        )
        base_scores: list[float] = []
        adapter_scores: list[float] = []
        for task in tasks:
            messages = [ChatMessage.user(task["prompt"])]
            base_response = await ctx.run.gateway.complete_text(
                RoleName.TARGET, base_role, messages
            )
            adapter_response = await ctx.run.gateway.complete_text(
                RoleName.ADAPTER, adapter_role, messages
            )
            base_scores.append(
                await self._judge_external(task["prompt"], base_response.text, ctx, settings)
            )
            adapter_scores.append(
                await self._judge_external(task["prompt"], adapter_response.text, ctx, settings)
            )

        signal = make_signal(
            "external_judge_total",
            base_scores,
            adapter_scores,
            direction="higher_better",
            seed=ctx.config.seed,
        )
        dashboard = build_dashboard(
            [signal],
            episodes=len(tasks),
            seeds=[ctx.config.seed],
            notes=["external calibration; totals are out of 25"],
        )
        ctx.emit_json("external", dashboard)
        ctx.emit_json(
            "evaluate_external_report",
            {"tasks": len(tasks), "delta": signal.delta, "improved": signal.improved},
        )
        ctx.emit_text("report", render_dashboard(dashboard))
        return StageResult.from_context(
            ctx,
            metrics={"tasks": float(len(tasks)), "delta": signal.delta},
        )


# ---------------------------------------------------------------------------
# learning curve — the data-volume instrument
# ---------------------------------------------------------------------------


class LearningCurveSettings(StageSettings):
    """Knobs for the learning-curve sweep.

    The curve re-trains on growing subsets of the *same* training set with the
    same hyperparameters as `train_sft`, so the only thing that varies is data
    volume. ``repeats`` draws per fraction average out subset-selection noise;
    Mahmood et al. (2022) warn that power-law extrapolation from few points is
    unstable, so the stage reports the curve, never a fitted forecast.
    """

    fractions: list[float] = Field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0],
    )
    repeats: int = Field(default=1, ge=1)


class LearningCurve(Stage):
    """Train on growing subsets of the SFT set and tabulate best val loss.

    The instrument behind GAP-093: whether more data would help is an empirical
    property of the curve, not a judgement call. In Hestness et al.'s (2017)
    terms it locates the corpus on the small-data / power-law / irreducible
    trajectory — a curve still falling at the full training set says data
    volume binds; a flat curve says it does not, and tuning or quality is the
    lever instead. Fine-tuning loss is expected to follow roughly
    ``L(D) ≈ C·|D|^-β + E`` (Zhang et al. 2024); the stage draws the curve and
    leaves the fit to the reader rather than extrapolating from four points.
    """

    spec = StageSpec(
        name="learning_curve",
        tier=Tier.C,
        requires_llm=False,
        lead=_EXT,
        roi="Medium",
        justification=(
            "Answers 'is more data the lever' directly; without it the "
            "over/under-fitting question is half-answered (GAP-093)."
        ),
        evidence=[
            "v12: 17 train records, 30-epoch ceiling, eval-loss min at ~8 epochs",
            "LIMA-style corpora sit orders of magnitude larger; the curve says "
            "which side of the knee this corpus is on",
        ],
        cost=CostProfile(compute="gpu", api_usd="none", maintenance="low"),
        inputs={"train": "jsonl", "val": "jsonl", "base_probe": "json"},
        outputs={"learning_curve": "json", "report": "text"},
        kill_criterion=("Drop if the full-size point stops reproducing train_sft's val loss."),
        gated_by=_gated("a full sweep costs one training run per point"),
    )
    settings_model = LearningCurveSettings

    async def run(self, ctx: StageContext) -> StageResult:
        """Train each subset and emit the curve."""
        settings = ctx.settings(self.settings_model)
        # Hyperparameters come from train_sft's block, not this stage's: the
        # curve must vary data volume *only*, and the full-size point has to be
        # comparable with the pipeline's actual training run.
        train_settings = ctx.settings(TrainSftSettings)
        probe = BaseProbe.model_validate(ctx.catalog.read_json("base_probe"))
        if not probe.trainable:
            raise StageError(
                "learning_curve",
                f"base model {probe.repo_id!r} is not trainable ({'; '.join(probe.notes)})",
            )
        train_records = list(ctx.catalog.stream("train", SFTRecord))
        val_entry = ctx.catalog.get("val")
        val_dir = str(Path(val_entry.path).parent)
        backend = ctx.run.backends.training()

        points: list[JsonObject] = []
        for fraction, repeat in _sweep(settings.fractions, settings.repeats):
            subset = _subset(train_records, fraction, repeat, ctx.config.seed)
            subset_dir = ctx.store.directory / _point_name(fraction, repeat)
            subset_dir.mkdir(parents=True, exist_ok=True)
            sink = (subset_dir / "train.jsonl").open("w", encoding="utf-8")
            for record in subset:
                sink.write(record.model_dump_json() + "\n")
            sink.close()
            epochs = _epochs_for(len(subset), ctx.config.hardware, train_settings)
            request = TrainingRequest(
                base_model=probe.path or probe.repo_id,
                dataset_dir=str(subset_dir),
                output_dir=str(subset_dir / "adapter"),
                hardware=ctx.config.hardware,
                seed=ctx.config.seed,
                max_seq_len=ctx.config.hardware.max_seq_len,
                learning_rate=train_settings.learning_rate,
                epochs=epochs,
                lora_r=train_settings.lora_r,
                lora_alpha=train_settings.lora_alpha,
                lora_dropout=train_settings.lora_dropout,
                target_modules=train_settings.target_modules or probe.target_modules,
                target_parameters=probe.target_parameters,
                resume_from=None,
                max_steps=-1,
                save_steps=train_settings.save_steps,
                eval_dir=val_dir,
            )
            ctx.logger.info(
                "learning-curve point",
                fraction=fraction,
                repeat=repeat,
                n_train=len(subset),
                epochs=epochs,
            )
            result: TrainingResult = await backend.train(request)
            points.append(
                {
                    "fraction": fraction,
                    "repeat": repeat,
                    "n_train": len(subset),
                    "epochs": epochs,
                    "steps": result.steps,
                    "train_loss": result.train_loss,
                    # The best held-out loss is what load_best_model_at_end
                    # selected on — the curve's y-axis.
                    "best_eval_loss": result.metrics.get(
                        "best_eval_loss", result.metrics.get("eval_loss")
                    ),
                    "wall_time_s": result.wall_time_s,
                }
            )
            ctx.emit_json("learning_curve", {"points": points, "val_records": val_entry.ref.rows})

        # The one derived indicator: relative val-loss improvement between the
        # two largest trained sizes. A still-falling curve at the full size is
        # the signature of data-bound underfitting; a flat one says volume is
        # not the lever. Descriptive statistics, not an agent verdict — the
        # curve itself is the evidence.
        by_size = sorted(points, key=lambda point: (point["n_train"], point["fraction"]))
        note = ""
        if len(by_size) >= 2:
            last, prev = by_size[-1], by_size[-2]
            prev_loss = prev.get("best_eval_loss")
            last_loss = last.get("best_eval_loss")
            # JsonObject values are JsonValue unions; the trainer's metrics dict
            # holds floats, and the isinstance narrowing is what tells mypy so.
            if isinstance(prev_loss, float) and isinstance(last_loss, float):
                improvement = (prev_loss - last_loss) / prev_loss
                note = (
                    f"val loss {'falls' if improvement > 0 else 'rises'} by "
                    f"{abs(improvement) * 100:.1f}% from n={prev['n_train']} to "
                    f"n={last['n_train']}"
                )
        body_lines = [
            "# Learning curve\n",
            "| fraction | repeat | n_train | epochs | steps | train loss "
            "| best val loss | wall s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for point in points:
            body_lines.append(
                f"| {point['fraction']:g} | {point['repeat']} | {point['n_train']} "
                f"| {point['epochs']:g} | {point['steps']} | {point['train_loss']:.4f} "
                f"| {_fmt_loss(point.get('best_eval_loss'))} | {point['wall_time_s']:.0f} |"
            )
        if note:
            body_lines += ["", f"- {note}."]
        body_lines += [
            "",
            "- best val loss is the checkpoint `load_best_model_at_end` selected;",
            "  hyperparameters are train_sft's, so only data volume varies.",
            "- a curve still falling at the largest n means data volume binds;",
            "  a flat curve means the corpus (of this quality mix) is not the lever.",
        ]
        ctx.emit_text("report", "\n".join(body_lines) + "\n")
        largest_n = max(
            int(point["n_train"]) for point in points if isinstance(point["n_train"], int)
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "points": float(len(points)),
                "largest_n": float(largest_n),
            },
        )


def _sweep(fractions: list[float], repeats: int) -> list[tuple[float, int]]:
    """Expand fractions x repeats into a deduplicated, ordered sweep."""
    valid = sorted({round(min(max(fraction, 0.01), 1.0), 4) for fraction in fractions})
    return [(fraction, repeat) for fraction in valid for repeat in range(1, repeats + 1)]


def _subset(records: list[SFTRecord], fraction: float, repeat: int, seed: int) -> list[SFTRecord]:
    """Draw a deterministic subset: seeded, sorted by id, at least one record."""
    count = max(1, min(len(records), round(len(records) * fraction)))
    rng = random.Random(f"{seed}:{fraction}:{repeat}")
    return sorted(rng.sample(records, count), key=lambda record: record.id)


def _point_name(fraction: float, repeat: int) -> str:
    return f"curve_f{round(fraction * 100):03d}_r{repeat}"


def _fmt_loss(loss: object) -> str:
    return f"{float(loss):.4f}" if isinstance(loss, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# capability probe — the forgetting instrument
# ---------------------------------------------------------------------------

_PROBE_PROMPT = "capability_judge"

# The probe's tasks are fixed general coding questions, deliberately NOT drawn
# from the corpus: they are the instrument, the same way export's parity probes
# are. They span code, explanation, tooling, and data shaping so a style-only
# adapter (one that learned "be terse") cannot pass them all by accident.
_PROBE_PROMPTS: tuple[str, ...] = (
    "Write a Python function `chunked(iterable, size)` that yields lists of at "
    "most `size` items from any iterable, including one-shot generators. Include "
    "a short example of use.",
    "Explain the difference between a process and a thread, and when you would "
    "choose one over the other. Keep it under 150 words.",
    "Write a bash one-liner that lists the 10 largest files under the current "
    "directory with their human-readable sizes, largest first.",
    "Write a Python function that flattens a nested dict into dot-separated "
    'keys, e.g. {"a": {"b": 1}} becomes {"a.b": 1}. Handle lists by index.',
    "What does `git rebase --onto A B C` do? Answer in under 100 words.",
    "Given tables users(id, name) and orders(id, user_id, created_at), write a "
    "SQL query returning each user's name, their order count, and the date of "
    "their most recent order.",
)


class CapabilityProbeSettings(StageSettings):
    """Knobs for the forgetting probe."""

    max_tokens: int = 1_024
    max_generation_chars: int = 20_000
    max_instruction_chars: int = 4_000


class CapabilityProbe(Stage):
    """Replay general coding prompts on base vs adapter: the forgetting check.

    The replay dashboard measures the target behaviour, which is also the only
    thing it can measure — a tuned model that got *worse at ordinary coding*
    while learning friction avoidance would be invisible to it (GAP-094). This
    stage replays a fixed set of corpus-independent prompts through both the
    base model and the served adapter from the same instruction, scores every
    completion with the shared judge, and reports the paired deltas with the
    same power floor as the replay: below six probes no 0.05-level claim is
    attainable, and the dashboard says so.
    """

    spec = StageSpec(
        name="capability_probe",
        tier=Tier.C,
        requires_llm=True,
        prompts=[_PROBE_PROMPT],
        roles=[RoleName.TARGET, RoleName.ADAPTER, RoleName.JUDGE],
        lead=_EXT,
        roi="Medium",
        justification=(
            "A regression check the replay cannot provide: only the target "
            "metric is measured today, so general degradation is invisible."
        ),
        evidence=[
            "v11r: one replay answer collapsed to 155 chars — style damage from "
            "tiny-data SFT is real and can reach ordinary answers",
            "Biderman et al. (TMLR 2024): forgetting arises from "
            "over-optimization; LoRA + modest lr mostly prevent it — 'mostly' "
            "is not a measurement",
        ],
        cost=CostProfile(compute="mixed", api_usd="low", maintenance="low"),
        inputs={"adapter": "json"},
        outputs={"probe_dashboard": "json", "probe": "jsonl", "report": "text"},
        kill_criterion=(
            "Drop if the probe's base scores stop reproducing across runs "
            "(the instrument itself has become noise)."
        ),
        gated_by=_gated(
            "needs the base and adapter servers up; run inside the replay phase "
            "or after `ft_pod.sh serve`"
        ),
    )
    settings_model = CapabilityProbeSettings

    _SIGNALS: tuple[tuple[str, Direction], ...] = (
        ("correctness", "higher_better"),
        ("clarity", "higher_better"),
        ("verbosity_chars", "lower_better"),
        ("format_valid", "higher_better"),
    )

    def _gen_role(
        self, ctx: StageContext, role: RoleName, settings: CapabilityProbeSettings
    ) -> LlmRoleConfig:
        try:
            base = ctx.config.role(role)
        except ConfigError as exc:
            raise StageError(
                "capability_probe",
                f"role {role.value!r} is not configured; serve the adapter and "
                "add the role (the replay phase's env is sufficient)",
            ) from exc
        return base.model_copy(
            update={"temperature": 0.0, "max_tokens": settings.max_tokens, "cache": False}
        )

    async def _generate(
        self,
        instruction: str,
        role: RoleName,
        config: LlmRoleConfig,
        ctx: StageContext,
        settings: CapabilityProbeSettings,
    ) -> str:
        messages = [ChatMessage(role="user", content=instruction)]
        response = await ctx.run.gateway.complete_text(role, config, messages, seed=ctx.config.seed)
        return response.text

    async def _judge(
        self,
        instruction: str,
        continuation: str,
        ctx: StageContext,
        settings: CapabilityProbeSettings,
    ) -> ProbeVerdict:
        try:
            role = ctx.config.role(RoleName.JUDGE)
        except ConfigError as exc:
            raise StageError("capability_probe", "the judge role is not configured") from exc
        # Greedy judging, as in the replay: a sampled judge would add noise to
        # the very comparison it is meant to measure.
        role = role.model_copy(update={"temperature": 0.0, "cache": True})
        return await ctx.run.gateway.complete_structured(
            RoleName.JUDGE,
            role,
            _PROBE_PROMPT,
            {
                "instruction": clip(instruction, settings.max_instruction_chars),
                "continuation": clip(continuation, settings.max_generation_chars),
            },
            ProbeVerdict,
        )

    async def _probe_one(
        self,
        instruction: str,
        index: int,
        ctx: StageContext,
        settings: CapabilityProbeSettings,
    ) -> JsonObject:
        """Generate and score both sides for one prompt; degrade, never abort.

        A failed generation or verdict on one side leaves that probe out of the
        signals but in the artifact with the error named — the compaction
        precedent. A probe that silently vanished would overstate the adapter.
        """
        base_role = self._gen_role(ctx, RoleName.TARGET, settings)
        adapter_role = self._gen_role(ctx, RoleName.ADAPTER, settings)
        row: JsonObject = {"id": f"probe_{index}", "instruction": instruction}
        for side, role, config in (
            ("base", RoleName.TARGET, base_role),
            ("adapter", RoleName.ADAPTER, adapter_role),
        ):
            try:
                continuation = await self._generate(instruction, role, config, ctx, settings)
                verdict = await self._judge(instruction, continuation, ctx, settings)
            except LlmError as exc:
                row[side] = {"error": str(exc)}
                continue
            row[side] = {
                "correctness": verdict.correctness,
                "clarity": verdict.clarity,
                "verbosity_chars": float(len(continuation)),
                "format_valid": _format_valid(continuation),
                "reason": verdict.reason,
            }
        return row

    async def run(self, ctx: StageContext) -> StageResult:
        """Probe both models on the fixed prompt set and emit paired deltas."""
        settings = ctx.settings(self.settings_model)
        rows = list(
            await asyncio.gather(
                *(
                    self._probe_one(instruction, index, ctx, settings)
                    for index, instruction in enumerate(_PROBE_PROMPTS, start=1)
                )
            )
        )

        baseline: dict[str, list[float]] = {name: [] for name, _dir in self._SIGNALS}
        tuned: dict[str, list[float]] = {name: [] for name, _dir in self._SIGNALS}
        scored = 0
        for row in rows:
            base, adapter = row["base"], row["adapter"]
            if not isinstance(base, dict) or not isinstance(adapter, dict):
                continue  # one side errored: kept in the artifact, out of the signals
            if "error" in base or "error" in adapter:
                continue
            scored += 1
            for name, _dir in self._SIGNALS:
                baseline[name].append(float(base[name]))  # type: ignore[arg-type]
                tuned[name].append(float(adapter[name]))  # type: ignore[arg-type]

        signals = [
            make_signal(
                name, baseline[name], tuned[name], direction=direction, seed=ctx.config.seed
            )
            for name, direction in self._SIGNALS
        ]
        # build_dashboard without a friction_avoidance signal leaves the
        # friction-drop headline at its default zero; the note says plainly
        # that the headline does not apply to this instrument.
        dashboard = build_dashboard(
            signals,
            episodes=scored,
            seeds=[ctx.config.seed],
            notes=[
                "capability probe: general-capability regression check on fixed "
                "corpus-independent prompts; the friction_drop headline does "
                "not apply here (no friction_avoidance axis)",
            ],
        )
        ctx.emit_json("probe_dashboard", dashboard)
        ctx.emit_jsonl("probe", rows)
        ctx.emit_text("report", render_dashboard(dashboard))
        if scored == 0:
            note = "every probe failed on at least one side; no paired signal"
            ctx.logger.warning(note)
            return StageResult.from_context(ctx, metrics={"scored": 0.0}, notes=[note])
        primary = dashboard.signal("correctness")
        return StageResult.from_context(
            ctx,
            metrics={
                "scored": float(scored),
                "base_correctness": primary.baseline if primary else 0.0,
                "adapter_correctness": primary.tuned if primary else 0.0,
            },
            notes=dashboard.notes,
        )


# ---------------------------------------------------------------------------
# preference pairs — the negative half the pipeline already mines
# ---------------------------------------------------------------------------


class PreferencePairsSettings(StageSettings):
    """Knobs for building chosen/rejected pairs from mined frictions."""

    # A rejected side must carry some substance to be a usable negative: an
    # empty or one-line failing turn teaches nothing a style prior does not.
    min_rejected_chars: int = Field(default=200, ge=1)
    max_pair_context_chars: int = Field(default=24_000, ge=1)
    max_pairs: int = Field(default=0, ge=0)  # 0 = all


def _pair_from_example(
    example: TrainingExample, event: FrictionEvent | None, settings: PreferencePairsSettings
) -> tuple[PreferencePair | None, str]:
    """Construct one pair from a recovery example, or return ``(None, reason)``.

    The rejected side isolates the assistant's failing turns: non-assistant
    messages inside the friction span (tool results, the user's correction)
    stay in the prompt, where they chronologically belong — the model must
    learn to act differently given the same state, not to avoid seeing it.
    """
    if event is None:
        return None, "event_not_found"
    if not example.context or not example.completion:
        return None, "empty_side"
    friction_turn = min(span.turn for span in event.evidence)
    if friction_turn >= len(example.context):
        return None, "friction_beyond_context"

    span = example.context[friction_turn:]
    prompt = [*example.context[:friction_turn], *(m for m in span if m.role != "assistant")]
    rejected = [m for m in span if m.role == "assistant"]

    def _chars(messages: list[Message]) -> int:
        return sum(len(message.content_text()) for message in messages)

    if not rejected or _chars(rejected) < settings.min_rejected_chars:
        return None, "rejected_too_thin"
    if not any(m.role == "assistant" for m in example.completion):
        return None, "chosen_without_assistant"
    # Trim the prompt from the oldest end to the pair budget; the most recent
    # state is what the chosen/rejected continuation hangs off.
    over = _chars(prompt) - settings.max_pair_context_chars
    while over > 0 and len(prompt) > 1:
        over -= len(prompt[0].content_text())
        prompt = prompt[1:]
    if not prompt:
        return None, "empty_prompt"
    return (
        PreferencePair(
            id=f"{example.id}/pair",
            source=example.source,
            prompt=prompt,
            chosen=list(example.completion),
            rejected=rejected,
            meta={
                "event_id": example.meta.get("event_id"),
                "leaf": event.leaf,
                "trigger": example.meta.get("trigger"),
                "session_id": example.meta.get("session_id"),
            },
        ),
        "paired",
    )


class BuildPreferencePairs(Stage):
    """Mine chosen/rejected pairs from recovery episodes (GAP-104).

    Positive-only SFT raises the probability of good continuations and can only
    approximate "avoid what went wrong" by style. The negative half is already
    in the artifacts: the friction event's evidence points at the assistant's
    actual failing turns, and the same session's recovery is the chosen side.
    KTO (Ethayarajh et al. 2024) trains from unpaired binary labels of exactly
    this shape — desirable/undesirable continuations under the same prompt —
    and outperforms DPO even at 72% less data when pairs are scarce. This stage
    emits the pairs; the trainer backend consumes them when one exists.
    """

    spec = StageSpec(
        name="build_preference_pairs",
        tier=Tier.C,
        requires_llm=False,
        lead=_EXT,
        roi="High",
        justification=(
            "The negative half of the data already exists as friction evidence; "
            "a preference objective is the only way to train *avoidance* rather "
            "than style (GAP-104)."
        ),
        evidence=[
            "v12: 54 recovery candidates judged — each carries a failing turn "
            "and a recovery under the same context",
            "KTO outperforms DPO from unpaired binary labels at a fraction of "
            "the data (Ethayarajh et al. 2024)",
        ],
        cost=CostProfile(compute="cpu", api_usd="none", maintenance="low"),
        inputs={"judged": "jsonl", "friction_events": "jsonl"},
        outputs={"preference_pairs": "jsonl", "report": "text"},
        kill_criterion=(
            "Drop if the pairs' rejected sides turn out to be duplicates of the "
            "chosen sides' style rather than distinct failures."
        ),
        gated_by=_gated("the pairs are inert until a preference-training backend exists"),
    )
    settings_model = PreferencePairsSettings

    async def run(self, ctx: StageContext) -> StageResult:
        """Join recovery examples with their friction events and emit pairs."""
        settings = ctx.settings(self.settings_model)
        events = {event.id: event for event in ctx.catalog.stream("friction_events", FrictionEvent)}
        sink = ctx.jsonl_sink("preference_pairs")
        reasons: dict[str, int] = {}
        pairs = 0
        tracks: dict[str, int] = {}
        for example in ctx.catalog.stream("judged", TrainingExample):
            if not example.kept:
                continue
            tracks[example.track] = tracks.get(example.track, 0) + 1
            if example.track != "recovery":
                continue
            if settings.max_pairs and pairs >= settings.max_pairs:
                break
            event = events.get(str(example.meta.get("event_id")))
            pair, reason = _pair_from_example(example, event, settings)
            reasons[reason] = reasons.get(reason, 0) + 1
            if pair is None:
                continue
            sink.write(pair)
            pairs += 1
        sink.close()

        lines = [
            "# Preference pairs\n",
            f"- pairs: {pairs}",
            "- judged tracks: "
            + (", ".join(f"{k}={v}" for k, v in sorted(tracks.items())) or "none"),
            "- skip reasons: "
            + (", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none"),
            "",
            "- rejected = the assistant's failing turns (friction evidence); chosen =",
            "  the same session's recovery; both hang off the pre-friction prompt.",
            "- inert until a preference-training backend (DPO/KTO) consumes them.",
        ]
        ctx.emit_text("report", "\n".join(lines) + "\n")
        return StageResult.from_context(
            ctx,
            metrics={
                "pairs": float(pairs),
                "judged_recovery": float(
                    sum(
                        1
                        for e in ctx.catalog.stream("judged", TrainingExample)
                        if e.track == "recovery" and e.kept
                    )
                ),
            },
        )


# ---------------------------------------------------------------------------
# registry helper
# ---------------------------------------------------------------------------

EXTENSION_STAGES: tuple[type[Stage], ...] = (
    Reduce,
    Attribute,
    FrictionCluster,
    Synthesize,
    RaftSrft,
    Rollout,
    TrainGrpo,
    EvaluateSimUser,
    EvaluateExternal,
    LearningCurve,
    CapabilityProbe,
    BuildPreferencePairs,
)


def register_extension_stages(registry: object) -> None:
    """Register every extension stage on ``registry``."""
    from frictiontune.pipeline.registry import StageRegistry

    if not isinstance(registry, StageRegistry):
        raise TypeError("register_extension_stages expects a StageRegistry")
    for stage_cls in EXTENSION_STAGES:
        registry.register(stage_cls)


__all__ = [
    "EXTENSION_STAGES",
    "Attribute",
    "EvaluateExternal",
    "EvaluateSimUser",
    "FrictionCluster",
    "RaftSrft",
    "Reduce",
    "Rollout",
    "Synthesize",
    "TrainGrpo",
    "register_extension_stages",
]
