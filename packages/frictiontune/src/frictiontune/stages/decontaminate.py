"""``decontaminate`` — make the friction-drop number trustworthy.

A held-out evaluation is worthless if its tasks leaked into training. The split
is a strict time split (future work is evaluated on, past work is trained on),
and four detection tiers run over it: T1 exact n-gram overlap and T2 syntactic
containment are deterministic measurements; T3/T4 semantic and task-level leaks
are judged by an agent over embedding-retrieved candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from frictiontune.config.models import RoleName
from frictiontune.core.errors import BackendUnavailableError, LlmError, StageError
from frictiontune.core.stats import cosine
from frictiontune.core.text import normalize_for_match, word_ngram_set
from frictiontune.domain.dataset import SFTRecord, TrainingExample
from frictiontune.domain.responses import ContaminationVerdict
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.stages.common import clip, render_transcript

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    embed_batch: int = Field(default=64, ge=1)
    max_text_chars: int = 20_000
    max_train_ngrams: int = 2_000_000
    # Bound on the contamination judge's prompt. `retrieve_k` texts of
    # `max_text_chars` each would otherwise exceed a server slot (and the 400 is
    # non-retryable), so the joined texts are clipped to this total.
    max_prompt_chars: int = 12_000


def _meta_index(
    ctx: StageContext, settings: DecontaminateSettings
) -> tuple[dict[str, tuple[str, int]], set[str]]:
    """Map example id to (rendered text, timestamp), plus the generated ids.

    Generated examples (synthesized coverage) carry no real timestamp, so they
    are returned separately: the time split is computed over real work only, and
    generated examples always train. Otherwise a synthetic backlog would shift
    the boundary and could push real work into the held-out evaluation the model
    was meant to learn from.
    """
    index: dict[str, tuple[str, int]] = {}
    generated: set[str] = set()
    for example in ctx.catalog.stream("judged", TrainingExample):
        if not example.kept:
            continue
        text = f"{render_transcript(example.context)}\n{render_transcript(example.completion)}"
        raw_time = example.meta.get("time_created", 0)
        time_created = int(raw_time) if isinstance(raw_time, (int, float)) else 0
        index[example.id] = (clip(text, settings.max_text_chars), time_created)
        if example.meta.get("generated") is True:
            generated.add(example.id)
    return index, generated


def _t1_overlap(eval_ngrams: set[str], train_ngrams: set[str]) -> float:
    if not eval_ngrams:
        return 0.0
    return len(eval_ngrams & train_ngrams) / len(eval_ngrams)


def _time_split(
    records: Sequence[SFTRecord],
    meta: Mapping[str, tuple[str, int]],
    generated: set[str],
    quantile: float,
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Split curated records into (train, eval) by a real-work time quantile.

    Generated (synthetic) examples always train and never evaluate: they have no
    real timestamp, so letting them into the sort would move the boundary and can
    push real work into the held-out evaluation the model was meant to learn
    from. The boundary is a quantile of the *real* records only.

    Records are assigned by their source **session**, not one by one: every
    episode of a session shares the user's task, tools, and context, so a record
    of one episode is not held out from a sibling episode of the same session.
    Splitting inside a session would train on the context of a task the model is
    then evaluated on, and — on a single-user corpus — leave a held-out set that
    the contamination tiers drop entirely (GAP-047). Whole sessions fall on one
    side; a missing session falls back to the record id so nothing merges.

    The held-out side is taken **from the newest session backwards** until it
    holds about ``1 - quantile`` of the real records. Walking the groups
    oldest-first and stopping at the boundary looks equivalent and is not: a
    large, newest session that straddles the boundary would go wholly to *train*,
    and the held-out set would come out empty. That is not hypothetical — the
    first run after the good-work track landed put 9 of 32 records in one newest
    session, the boundary fell at 25, the accumulated total reached 23 before it,
    and every record went to train.
    """
    ordered = sorted(records, key=lambda record: (meta.get(record.id, ("", 0))[1], record.id))
    generated_records = [record for record in ordered if record.id in generated]
    real_records = [record for record in ordered if record.id not in generated]

    # Group in time order; a group is one session. Fall back to the record id so
    # a missing session can never merge unrelated records.
    groups: dict[str, list[SFTRecord]] = {}
    for record in real_records:
        session = record.meta.get("session_id") or record.id
        groups.setdefault(str(session), []).append(record)

    target_eval = len(real_records) - int(quantile * len(real_records))
    held_out: set[str] = set()
    held_count = 0
    for session, group in reversed(list(groups.items())):
        if held_count >= target_eval:
            break
        held_out.add(session)
        held_count += len(group)
    eval_real = [
        record
        for record in real_records
        if str(record.meta.get("session_id") or record.id) in held_out
    ]
    held_ids = {record.id for record in eval_real}
    train_real = [record for record in real_records if record.id not in held_ids]
    return [*generated_records, *train_real], eval_real


def _t2_containment(eval_norm: str, train_norms: list[str]) -> bool:
    if not eval_norm:
        return False
    return any(eval_norm in train or train in eval_norm for train in train_norms if train)


def _contamination_report(
    *,
    total: int,
    generated_train: int,
    train: int,
    surviving: int,
    dropped: int,
    quantile: float,
    reasons: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    """Build the contamination report JSON and its markdown body.

    An eval set that ends up empty while real (non-generated) training records
    exist is called out explicitly: it makes the held-out measure vacuous, and the
    downstream ``evaluate_replay`` will refuse to run without examples.
    """
    eval_empty = surviving == 0 and train > generated_train
    report: dict[str, object] = {
        "total": total,
        "generated_train": generated_train,
        "train": train,
        "eval": surviving,
        "eval_empty": eval_empty,
        "dropped_contaminated": dropped,
        "time_split_quantile": quantile,
        "reasons": dict(reasons),
    }
    body = (
        "# Decontaminate\n\n"
        f"- total: {total}\n"
        f"- train (past): {train}\n"
        f"- eval (future): {surviving}\n"
        f"- dropped as contaminated: {dropped}\n"
    )
    if eval_empty:
        body += "\n- **WARNING: eval set is empty** — the held-out measure is vacuous.\n"
    return report, body


class Decontaminate(Stage):
    """Split curated data by time and remove contaminated evaluation items."""

    spec = StageSpec(
        name="decontaminate",
        tier=Tier.A,
        requires_llm=True,
        uses_evaluation=True,
        prompts=["decontaminate"],
        roles=[RoleName.JUDGE],
        lead=Lead.CORE,
        roi="High",
        justification="Without it the headline friction drop is untrustworthy.",
        evidence=["time split + T1-T4 tiers over embedding-retrieved candidates"],
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
        settings: DecontaminateSettings,
    ) -> bool:
        role = ctx.config.role(RoleName.JUDGE)
        try:
            verdict: ContaminationVerdict = await ctx.run.gateway.complete_structured(
                RoleName.JUDGE,
                role,
                _DECON_PROMPT,
                {
                    "eval_text": clip(eval_text, settings.max_prompt_chars),
                    "train_texts": clip("\n\n---\n\n".join(train_texts), settings.max_prompt_chars),
                },
                ContaminationVerdict,
            )
        except (LlmError, StageError):
            # A failed contamination judgement is treated as contaminated, the
            # conservative direction: training on a leak invalidates the eval,
            # which is worse than dropping one fair test. It must never abort the
            # stage.
            return True
        return verdict.contaminated

    async def run(self, ctx: StageContext) -> StageResult:
        """Apply the time split and every enabled contamination tier."""
        settings = ctx.settings(self.settings_model)
        eval_cfg = ctx.config.evaluation

        meta, generated = _meta_index(ctx, settings)
        records = list(ctx.catalog.stream("curated", SFTRecord))
        train_records, eval_records = _time_split(
            records, meta, generated, eval_cfg.time_split_quantile
        )

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

        report, body = _contamination_report(
            total=len(records),
            generated_train=sum(1 for record in train_records if record.id in generated),
            train=len(train_records),
            surviving=len(surviving),
            dropped=len(dropped),
            quantile=eval_cfg.time_split_quantile,
            reasons=reasons,
        )
        # An empty evaluation set silently disables the only honest measure of
        # whether tuning helped; downstream `evaluate_replay` will refuse to run.
        # Surface it here, where the cause (a drop) is still visible.
        if report["eval_empty"]:
            ctx.logger.warning(
                "evaluation set is empty: every held-out record was dropped as contaminated",
                train=len(train_records),
                dropped=len(dropped),
            )
        ctx.emit_json("contamination_report", report)
        ctx.emit_text("report", body)
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

            if await self._semantic_check(text, candidates[: settings.retrieve_k], ctx, settings):
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
