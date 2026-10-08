"""``build_sft`` — render curated examples into masked, model-ready records.

Template and mask correctness decide whether training works at all: labels are
``-100`` on everything that is context or environment output, so the loss is
computed only over the assistant behaviour we want to teach. Whether reasoning is
trained or masked is a first-class, configurable policy here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.core.errors import StageError
from frictiontune.domain.dataset import SFTRecord, TokenStats, TrainingExample
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings
from frictiontune.sft.masking import IGNORE_INDEX, fit_and_label
from frictiontune.sft.renderer import render_chat, to_chat_messages
from frictiontune.sft.tokenizer import Tokenizer, build_tokenizer

if TYPE_CHECKING:
    from frictiontune.pipeline.context import StageContext

_HISTOGRAM_EDGES = (512, 1024, 2048, 4096, 8192, 16384, 32768)


class BuildSftSettings(StageSettings):
    """Knobs for tokenization, masking, and the context window."""

    tokenizer_path: str | None = None
    prefer_hf: bool = True
    trust_remote_code: bool = False
    allow_fallback_tokenizer: bool = False
    include_reasoning_context: bool = True
    train_reasoning: bool = False
    max_seq_len: int = 8_192
    batch_size: int = 64


def _histogram_key(length: int) -> str:
    previous = 0
    for edge in _HISTOGRAM_EDGES:
        if length <= edge:
            return f"{previous}-{edge}"
        previous = edge
    return f"{previous}+"


def build_record(
    example: TrainingExample,
    tokenizer: Tokenizer,
    *,
    settings: BuildSftSettings,
    tokenizer_kind: str,
) -> tuple[SFTRecord, bool, int]:
    """Render one example into a tokenized record; report truncation and trained tokens."""
    context = to_chat_messages(
        example.context, include_reasoning=settings.include_reasoning_context
    )
    completion = to_chat_messages(example.completion, include_reasoning=settings.train_reasoning)
    messages = [*context, *completion]
    flags = [False] * len(context) + [message.role == "assistant" for message in completion]
    rendered = render_chat(tokenizer, messages, flags)
    fitted = fit_and_label(rendered, settings.max_seq_len)
    trained = sum(1 for label in fitted.labels if label != IGNORE_INDEX)
    record = SFTRecord(
        id=example.id,
        source=example.source,
        track=example.track,
        input_ids=fitted.input_ids,
        labels=fitted.labels,
        meta={
            **example.meta,
            "truncated": fitted.truncated,
            "tokenizer": tokenizer_kind,
            "trained_tokens": trained,
        },
    )
    return record, fitted.truncated, trained


class BuildSft(Stage):
    """Tokenize judged examples into a masked SFT dataset."""

    spec = StageSpec(
        name="build_sft",
        tier=Tier.A,
        lead=Lead.CORE,
        roi="High",
        justification="Template and mask correctness decide whether training works at all.",
        evidence=["91.2% of assistant messages carry reasoning (p90 5.4 KB)"],
        cost=CostProfile(compute="cpu", api_usd="none", maintenance="medium"),
        inputs={"judged": "jsonl"},
        outputs={"sft": "jsonl", "token_stats": "json"},
        kill_criterion="Drop only if a standard trainer handles masking natively and equally.",
    )
    settings_model = BuildSftSettings

    async def run(self, ctx: StageContext) -> StageResult:
        """Render every kept example and emit the SFT dataset and token report."""
        settings = ctx.settings(self.settings_model)
        tokenizer, used_fallback = build_tokenizer(
            settings.tokenizer_path,
            prefer_hf=settings.prefer_hf,
            trust_remote_code=settings.trust_remote_code,
        )
        if used_fallback and not settings.allow_fallback_tokenizer:
            raise StageError(
                "build_sft",
                "the base model tokenizer is unavailable, so token ids would be "
                "meaningless hashes; set stages.build_sft.tokenizer_path to the "
                "base model tokenizer, or set allow_fallback_tokenizer: true to "
                "build a throwaway dataset (e.g. for tests)",
            )
        tokenizer_kind = "fallback" if used_fallback else "huggingface"
        sink = ctx.jsonl_sink("sft")
        stats = TokenStats()
        lengths: list[int] = []
        skipped_empty = 0

        for example in ctx.catalog.stream("judged", TrainingExample):
            if not example.kept:
                continue
            record, truncated, trained = build_record(
                example, tokenizer, settings=settings, tokenizer_kind=tokenizer_kind
            )
            # A record with no trained token (e.g. a reasoning-only completion
            # when reasoning is masked) contributes no gradient and can poison a
            # batch; drop it and count it.
            if trained == 0:
                skipped_empty += 1
                continue
            sink.write(record)
            length = len(record.input_ids)
            stats.examples += 1
            stats.total_tokens += length
            stats.masked_tokens += length - trained
            stats.trained_tokens += trained
            stats.truncated += int(truncated)
            lengths.append(length)
            key = _histogram_key(length)
            stats.histogram[key] = stats.histogram.get(key, 0) + 1
        sink.close()

        if lengths:
            stats.max_tokens = max(lengths)
            stats.min_tokens = min(lengths)
            stats.mean_tokens = sum(lengths) / len(lengths)

        ctx.emit_json("token_stats", stats)
        ctx.emit_text(
            "report.md",
            "# Build SFT\n\n"
            f"- examples: {stats.examples}\n"
            f"- skipped (no trained tokens): {skipped_empty}\n"
            f"- total tokens: {stats.total_tokens}\n"
            f"- trained tokens: {stats.trained_tokens} "
            f"({stats.train_ratio() * 100:.1f}%)\n"
            f"- truncated: {stats.truncated}\n"
            f"- tokenizer: {tokenizer_kind}\n"
            f"- train reasoning: {settings.train_reasoning}\n",
        )
        notes = ["used fallback tokenizer; counts are approximate"] if used_fallback else []
        return StageResult.from_context(
            ctx,
            metrics={
                "examples": float(stats.examples),
                "skipped_empty": float(skipped_empty),
                "total_tokens": float(stats.total_tokens),
                "trained_tokens": float(stats.trained_tokens),
                "truncated": float(stats.truncated),
                "train_ratio": stats.train_ratio(),
            },
            notes=notes,
        )
