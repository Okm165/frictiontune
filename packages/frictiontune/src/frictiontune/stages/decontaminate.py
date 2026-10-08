"""``decontaminate`` — make the friction-drop number trustworthy.

A held-out evaluation is worthless if its tasks leaked into training. The split
is a strict time split (future work is evaluated on, past work is trained on),
and four detection tiers run over it: T1 exact n-gram overlap and T2 syntactic
containment are deterministic measurements; T3/T4 semantic and task-level leaks
are judged by an agent over embedding-retrieved candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.config.models import RoleName
from frictiontune.core.errors import BackendUnavailableError
from frictiontune.core.stats import cosine
from frictiontune.core.text import normalize_for_match, word_ngram_set
from frictiontune.domain.dataset import SFTRecord, TrainingExample
from frictiontune.domain.responses import ContaminationVerdict
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from frictiontune.pipeline.context import StageContext

_DECON_PROMPT = "decontaminate"


class DecontaminateSettings(StageSettings):
    """Knobs for the split and the contamination tiers."""

    t1_overlap: float = 0.5
    retrieve_k: int = 5
    # A retrieval recall bound only, like curate's: candidates below it are not
    # shown to the contamination judge. The default admits every positively
    # correlated neighbour so semantic contamination reaches the judge; raise it
    # to trade recall for fewer judge calls. It never decides the drop itself.
    retrieval_floor: float = 0.0
    embed_batch: int = 64
    max_text_chars: int = 20_000
    max_train_ngrams: int = 2_000_000


def _meta_index(ctx: StageContext, settings: DecontaminateSettings) -> dict[str, tuple[str, int]]:
    """Map example id to (rendered text, timestamp) from the judged artifact."""
    index: dict[str, tuple[str, int]] = {}
    for example in ctx.catalog.stream("judged", TrainingExample):
        if not example.kept:
            continue
        text = f"{render_transcript(example.context)}\n{render_transcript(example.completion)}"
        raw_time = example.meta.get("time_created", 0)
        time_created = int(raw_time) if isinstance(raw_time, (int, float)) else 0
        index[example.id] = (clip(text, settings.max_text_chars), time_created)
    return index


def _t1_overlap(eval_ngrams: set[str], train_ngrams: set[str]) -> float:
    if not eval_ngrams:
        return 0.0
    return len(eval_ngrams & train_ngrams) / len(eval_ngrams)


def _t2_containment(eval_norm: str, train_norms: list[str]) -> bool:
    if not eval_norm:
        return False
    return any(eval_norm in train or train in eval_norm for train in train_norms if train)


class Decontaminate(Stage):
    """Split curated data by time and remove contaminated evaluation items."""

    spec = StageSpec(
        name="decontaminate",
        tier=Tier.A,
        requires_llm=True,
        lead=Lead.CORE,
        roi="High",
        justification="Without it the headline friction drop is untrustworthy.",
        evidence=["time split + T1-T4; lucky-pass flag"],
        cost=CostProfile(compute="mixed", api_usd="medium", maintenance="medium"),
        inputs={"curated": "jsonl", "judged": "jsonl"},
        outputs={"train": "jsonl", "eval": "jsonl", "contamination_report": "json"},
        kill_criterion="Relax tiers only when the eval set is demonstrably clean another way.",
    )
    settings_model = DecontaminateSettings

    async def _semantic_check(
        self,
        eval_text: str,
        train_texts: list[str],
        ctx: StageContext,
    ) -> bool:
        role = ctx.config.role(RoleName.JUDGE)
        verdict: ContaminationVerdict = await ctx.run.gateway.complete_structured(
            RoleName.JUDGE,
            role,
            _DECON_PROMPT,
            {"eval_text": eval_text, "train_texts": "\n\n---\n\n".join(train_texts)},
            ContaminationVerdict,
        )
        return verdict.contaminated

    async def run(self, ctx: StageContext) -> StageResult:
        """Apply the time split and every enabled contamination tier."""
        settings = ctx.settings(self.settings_model)
        eval_cfg = ctx.config.evaluation

        meta = _meta_index(ctx, settings)
        records = list(ctx.catalog.stream("curated", SFTRecord))
        records.sort(key=lambda record: meta.get(record.id, ("", 0))[1])
        boundary = int(eval_cfg.time_split_quantile * len(records)) if records else 0
        train_records = records[:boundary]
        eval_records = records[boundary:]

        train_texts = [meta.get(record.id, ("", 0))[0] for record in train_records]
        train_norms = [normalize_for_match(text) for text in train_texts]
        train_ngrams: set[str] = set()
        for text in train_texts:
            if len(train_ngrams) >= settings.max_train_ngrams:
                break
            train_ngrams |= word_ngram_set(text, eval_cfg.ngram_size)

        dropped: list[str] = []
        reasons: dict[str, str] = {}
        semantic_enabled = eval_cfg.t3_semantic or eval_cfg.t4_task_level
        flagged_tier = eval_cfg.t1_exact_ngram or eval_cfg.t2_syntactic

        # The deterministic tiers MEASURE leak evidence; the agent decides whether
        # it is contamination. Nothing is dropped here.
        audit: list[tuple[SFTRecord, str, float, bool]] = []
        for record in eval_records:
            text = meta.get(record.id, ("", 0))[0]
            overlap = (
                _t1_overlap(word_ngram_set(text, eval_cfg.ngram_size), train_ngrams)
                if eval_cfg.t1_exact_ngram
                else 0.0
            )
            contained = eval_cfg.t2_syntactic and _t2_containment(
                normalize_for_match(text), train_norms
            )
            audit.append((record, text, overlap, contained))

        if semantic_enabled or flagged_tier:
            surviving = await self._agent_decide(
                audit,
                train_texts,
                train_norms,
                settings,
                ctx,
                dropped,
                reasons,
                semantic_enabled=semantic_enabled,
            )
        else:
            surviving = [record for record, *_ in audit]

        train_sink = ctx.jsonl_sink("train")
        for record in train_records:
            train_sink.write(record)
        train_sink.close()
        eval_sink = ctx.jsonl_sink("eval")
        for record in surviving:
            eval_sink.write(record)
        eval_sink.close()

        report = {
            "total": len(records),
            "train": len(train_records),
            "eval": len(surviving),
            "dropped_contaminated": len(dropped),
            "time_split_quantile": eval_cfg.time_split_quantile,
            "reasons": reasons,
        }
        ctx.emit_json("contamination_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Decontaminate\n\n"
            f"- total: {len(records)}\n"
            f"- train (past): {len(train_records)}\n"
            f"- eval (future): {len(surviving)}\n"
            f"- dropped as contaminated: {len(dropped)}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "total": float(len(records)),
                "train": float(len(train_records)),
                "eval": float(len(surviving)),
                "dropped_contaminated": float(len(dropped)),
            },
        )

    async def _agent_decide(
        self,
        audit: list[tuple[SFTRecord, str, float, bool]],
        train_texts: list[str],
        train_norms: list[str],
        settings: DecontaminateSettings,
        ctx: StageContext,
        dropped: list[str],
        reasons: dict[str, str],
        *,
        semantic_enabled: bool,
    ) -> list[SFTRecord]:
        """Ask the agent whether each flagged or semantically-nearest eval item leaks.

        The deterministic tiers only *retrieve and measure*; the drop is the
        agent's call. Embeddings retrieve the nearest training examples, and T2
        containment contributes the specific containing text. Only when an eval
        item is deterministically flagged and no candidate text can be retrieved
        (no embedder) do we fall back to dropping, because training on a leak is
        worse than losing an eval item.
        """
        if not audit:
            return []
        try:
            embedder = ctx.run.backends.embedder()
        except BackendUnavailableError:
            ctx.logger.warning(
                "no embedder: deterministic leak signals will drop without agent review"
            )
            embedder = None

        train_vectors: list[list[float]] = []
        if embedder is not None and train_texts:
            for start in range(0, len(train_texts), settings.embed_batch):
                train_vectors.extend(
                    await embedder.embed(train_texts[start : start + settings.embed_batch])
                )

        kept: list[SFTRecord] = []
        for record, text, overlap, contained in audit:
            flagged = contained or overlap >= settings.t1_overlap
            if not flagged and not semantic_enabled:
                kept.append(record)
                continue

            candidates: list[str] = []
            if embedder is not None and train_vectors:
                vector = (await embedder.embed([text]))[0]
                scored = sorted(
                    (
                        (cosine(vector, train_vector), index)
                        for index, train_vector in enumerate(train_vectors)
                    ),
                    reverse=True,
                )
                candidates = [
                    train_texts[index]
                    for score, index in scored[: settings.retrieve_k]
                    if score >= settings.retrieval_floor
                ]
            if contained:
                candidates = _dedupe(candidates + _containing(train_norms, train_texts, text))

            if not candidates:
                if flagged:
                    dropped.append(record.id)
                    reasons[record.id] = "t2-syntactic" if contained else f"t1-ngram:{overlap:.2f}"
                else:
                    kept.append(record)
                continue

            if await self._semantic_check(text, candidates[: settings.retrieve_k], ctx):
                dropped.append(record.id)
                reasons[record.id] = "t3/t4-semantic"
                continue
            kept.append(record)
        return kept


def _containing(train_norms: list[str], train_texts: list[str], text: str) -> list[str]:
    """Return train texts whose normalized form contains (or is contained by) the eval text."""
    normalized = normalize_for_match(text)
    if not normalized:
        return []
    return [
        train_texts[index]
        for index, norm in enumerate(train_norms)
        if norm and (normalized in norm or norm in normalized)
    ]


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
