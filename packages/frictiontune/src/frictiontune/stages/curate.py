"""``curate`` — deduplicate the SFT pool without destroying diversity.

Two passes: an exact byte-hash pass (pure plumbing) and an agentic semantic pass.
Embeddings only *retrieve* candidate pairs; a dedup agent decides whether two
examples teach the same lesson, so there is no shingle or cosine threshold making
the semantic call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from frictiontune.config.models import RoleName
from frictiontune.core.errors import BackendUnavailableError
from frictiontune.core.hashing import content_hash
from frictiontune.core.stats import cosine
from frictiontune.domain.dataset import SFTRecord, TrainingExample
from frictiontune.domain.responses import DedupVerdict
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from frictiontune.pipeline.context import StageContext

_DEDUP_PROMPT = "dedup"


class CurateSettings(StageSettings):
    """Knobs for exact and semantic deduplication."""

    exact_dedup: bool = True
    near_dedup: bool = True
    # A retrieval recall bound only: candidates below it are not shown to the
    # dedup agent. The default admits every positively-correlated neighbour so
    # paraphrases reach the agent; raise it to trade recall for fewer agent calls.
    # It never makes the duplicate decision itself.
    retrieval_floor: float = 0.0
    retrieve_k: int = 5
    compare_window: int = 64
    embed_batch: int = Field(default=64, ge=1)
    max_text_chars: int = 20_000


def _example_text(example: TrainingExample, limit: int) -> str:
    context = clip(render_transcript(example.context), limit)
    completion = clip(render_transcript(example.completion), limit)
    return f"CONTEXT:\n{context}\n\nCOMPLETION:\n{completion}"


class Curate(Stage):
    """Deduplicate candidates and emit the curated SFT pool."""

    spec = StageSpec(
        name="curate",
        tier=Tier.A,
        requires_llm=True,
        prompts=["dedup"],
        roles=[RoleName.DEDUP],
        lead=Lead.CORE,
        roi="High",
        justification="Deduplication improves models; semantic dedup beats surface similarity.",
        evidence=["dedup 2107.06499"],
        cost=CostProfile(compute="mixed", api_usd="medium", maintenance="medium"),
        inputs={"judged": "jsonl", "sft": "jsonl"},
        outputs={"curated": "jsonl", "curate_report": "json"},
        kill_criterion="Keep only exact dedup if the agentic pass removes too little.",
    )
    settings_model = CurateSettings

    def _collect(
        self, ctx: StageContext, settings: CurateSettings
    ) -> tuple[list[tuple[str, TrainingExample, str]], int]:
        """Collect kept examples with a text view, dropping exact duplicates."""
        seen: dict[str, str] = {}
        items: list[tuple[str, TrainingExample, str]] = []
        exact_dropped = 0
        for example in ctx.catalog.stream("judged", TrainingExample):
            if not example.kept:
                continue
            text = _example_text(example, settings.max_text_chars)
            digest = content_hash(text)
            if settings.exact_dedup and digest in seen:
                exact_dropped += 1
                continue
            seen[digest] = example.id
            items.append((example.id, example, text))
        return items, exact_dropped

    async def _embed(
        self, texts: list[str], ctx: StageContext, settings: CurateSettings
    ) -> list[list[float]]:
        embedder = ctx.run.backends.embedder()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embed_batch):
            vectors.extend(await embedder.embed(texts[start : start + settings.embed_batch]))
        return vectors

    async def _semantic_dedup(
        self,
        items: list[tuple[str, TrainingExample, str]],
        vectors: list[list[float]],
        ctx: StageContext,
        settings: CurateSettings,
    ) -> set[str]:
        """Return the ids to drop according to the dedup agent."""
        role = ctx.config.role(RoleName.DEDUP)
        dropped: set[str] = set()
        kept_indices: list[int] = []

        for current, vector in zip(range(len(items)), vectors, strict=True):
            current_id, _example, current_text = items[current]
            window = kept_indices[-settings.compare_window :]
            scored = sorted(
                ((cosine(vector, vectors[index]), index) for index in window),
                reverse=True,
            )
            candidates = [
                index
                for score, index in scored[: settings.retrieve_k]
                if score >= settings.retrieval_floor
            ]
            is_duplicate = False
            for index in candidates:
                existing_id, _existing, existing_text = items[index]
                verdict: DedupVerdict = await ctx.run.gateway.complete_structured(
                    RoleName.DEDUP,
                    role,
                    _DEDUP_PROMPT,
                    {"a": existing_text, "b": current_text},
                    DedupVerdict,
                )
                if not verdict.duplicate:
                    continue
                if verdict.keep == "b":
                    dropped.add(existing_id)
                    # `index` was the incumbent; the agent prefers the current
                    # item, so stop comparing later items against the one we are
                    # about to drop. The replacement must join the kept set, or
                    # later near-duplicates would never be compared against it.
                    kept_indices.remove(index)
                    kept_indices.append(current)
                else:
                    dropped.add(current_id)
                is_duplicate = True
                break
            if not is_duplicate:
                kept_indices.append(current)
        return dropped

    async def run(self, ctx: StageContext) -> StageResult:
        """Deduplicate and emit curated records."""
        settings = ctx.settings(self.settings_model)
        items, exact_dropped = self._collect(ctx, settings)
        dropped: set[str] = set()

        if settings.near_dedup and items:
            texts = [text for _id, _example, text in items]
            try:
                vectors = await self._embed(texts, ctx, settings)
            except BackendUnavailableError:
                vectors = []
            if vectors:
                dropped = await self._semantic_dedup(items, vectors, ctx, settings)

        keep_ids = {identifier for identifier, _example, _text in items} - dropped
        sink = ctx.jsonl_sink("curated")
        written = 0
        for record in ctx.catalog.stream("sft", SFTRecord):
            if record.id in keep_ids:
                sink.write(record)
                written += 1
        sink.close()

        report = {
            "collected": len(items),
            "exact_dropped": exact_dropped,
            "semantic_dropped": len(dropped),
            "kept": written,
        }
        ctx.emit_json("curate_report", report)
        ctx.emit_text(
            "report",
            "# Curate\n\n"
            f"- collected (after exact dedup): {len(items)}\n"
            f"- semantic duplicates dropped: {len(dropped)}\n"
            f"- records written: {written}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "collected": float(len(items)),
                "semantic_dropped": float(len(dropped)),
                "kept": float(written),
            },
        )
