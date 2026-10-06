"""Evidence-gated extension stages (Tier C).

These are the demoted steps of the plan. Each is real and runnable, but only
after the core pipeline has produced evidence that justifies its cost; the
orchestrator refuses to run a Tier C stage unless ``allow_gated`` is set.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import Field

from frictiontune.backends.models import (
    GrpoRequest,
    RolloutRequest,
    RolloutResult,
    TrainingResult,
)
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import StageError
from frictiontune.domain.dataset import TrainingExample
from frictiontune.domain.episodes import Episode, ReducedSpan
from frictiontune.domain.friction import FrictionEvent
from frictiontune.domain.ids import example_id
from frictiontune.domain.profiles import Attribution, EventAttribution, PreferenceProfile
from frictiontune.domain.responses import (
    FrictionClusters,
    ReduceSummary,
    SimulatedUserTurn,
    SyntheticBatch,
)
from frictiontune.domain.trace import Message, TextPart
from frictiontune.llm.messages import ChatMessage
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.reporting.dashboard import build_dashboard, make_signal, relative_drop
from frictiontune.reporting.report import render_dashboard
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.pipeline.context import StageContext

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


def _read_events(ctx: StageContext) -> dict[str, list[FrictionEvent]]:
    grouped: dict[str, list[FrictionEvent]] = defaultdict(list)
    for event in ctx.catalog.stream("friction_events", FrictionEvent):
        grouped[event.episode_id].append(event)
    return grouped


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


class ReduceSettings(StageSettings):
    """Knobs for episode-span reduction."""

    min_chars: int = 40_000
    chunk_turns: int = 40
    batch_size: int = 16


class Reduce(Stage):
    """Map-reduce compression of long episodes before analysis."""

    spec = StageSpec(
        name="reduce",
        tier=Tier.C,
        requires_llm=True,
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
                "transcript": render_transcript(chunk),
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

    batch_size: int = 32
    max_transcript_chars: int = 100_000


class Attribute(Stage):
    """Localise the earliest decisive step for each friction event."""

    spec = StageSpec(
        name="attribute",
        tier=Tier.C,
        requires_llm=True,
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
        grouped = _read_events(ctx)
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


class FrictionCluster(Stage):
    """Group friction events into behaviour clusters (the Preference Profile)."""

    spec = StageSpec(
        name="friction_cluster",
        tier=Tier.C,
        requires_llm=True,
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
                {"events": self._render(events)},
                FrictionClusters,
            )
            known = {event.id for event in events}
            covered = sum(
                1
                for dimension in clusters.clusters
                for event_id in dimension.event_ids
                if event_id in known
            )
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


class Synthesize(Stage):
    """Generate synthetic friction-avoiding examples per behaviour dimension."""

    spec = StageSpec(
        name="synthesize",
        tier=Tier.C,
        requires_llm=True,
        lead=_EXT,
        roi="Medium",
        justification=(
            "Rare behaviours need generated coverage; synthesis is optional"
            " while real recoveries remain."
        ),
        evidence=["RAFT/SRFT synthesis; PRIME"],
        cost=CostProfile(compute="llm", api_usd="medium", maintenance="medium"),
        inputs={"preference_profile": "json", "judged": "jsonl"},
        outputs={"synthetic": "jsonl", "synthesize_report": "json"},
        kill_criterion=(
            "Drop if synthetic examples fail the judge at a higher rate than mined ones."
        ),
        gated_by=_gated("mined recoveries may already cover the needed behaviours"),
    )
    settings_model = SynthesizeSettings

    def _seed_index(self, ctx: StageContext) -> dict[str, str]:
        seeds: dict[str, str] = {}
        for example in ctx.catalog.stream("judged", TrainingExample):
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
                    "seeds": "\n\n---\n\n".join(seed_texts) or "(no seeds available)",
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

    def _clean_episode_example(self, episode: Episode, index: int) -> TrainingExample | None:
        assistant_positions = [
            position
            for position, message in enumerate(episode.messages)
            if message.role == "assistant"
        ]
        if len(assistant_positions) < self._min_turns:
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
        self._min_turns = settings.min_assistant_turns
        grouped = _read_events(ctx)
        sink = ctx.jsonl_sink("sft_raft")
        produced = 0
        seen = 0
        for episode in ctx.catalog.stream("episodes", Episode):
            if episode.id in grouped:
                continue
            if seen >= settings.max_episodes:
                break
            seen += 1
            example = self._clean_episode_example(episode, produced)
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
        inputs={"train": "jsonl", "judged": "jsonl", "adapter": "json", "base_probe": "json"},
        outputs={"grpo_result": "json", "train_grpo_report": "json"},
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
    """Drive a simulated user against base and tuned models, counting turns."""

    spec = StageSpec(
        name="evaluate_sim_user",
        tier=Tier.C,
        requires_llm=True,
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
        except KeyError as exc:
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
            history.append(ChatMessage.assistant(response.text))
            turns += 1
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

    def _load_tasks(self, ctx: StageContext) -> list[dict[str, str]]:
        from frictiontune.core.json import iter_jsonl

        tasks: list[dict[str, str]] = []
        for path_str in ctx.config.evaluation.external_corpora:
            from pathlib import Path

            path = Path(path_str)
            if not path.is_file():
                ctx.logger.warning("external corpus missing", path=path_str)
                continue
            for row in iter_jsonl(path):
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
                "accept_threshold": 0,
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

        base_role = ctx.config.role(RoleName.TARGET)
        adapter_role = ctx.config.role(RoleName.ADAPTER)
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
