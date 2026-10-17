"""``normalize`` — turn raw parts into trainable, de-identified messages.

Three transforms, each split into plumbing plus judgment:

* **compaction** — long tool outputs are trimmed by an agent that names the
  line ranges safe to drop; the plumbing copies every other line verbatim, so
  error text in the middle survives (head/tail truncation would cut it out);
* **schema inference** — tool JSON Schemas are inferred from observed calls by an
  agent, never by merging keys;
* **redaction** — candidate secrets are detected by regex (plumbing) across
  every string leaf of a message, including tool-call arguments, and then judged
  by an agent; over-redaction is a real cost, so the model decides.

Messages are streamed in batches with bounded concurrency and written to a JSONL
sink, so the whole corpus is never resident.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import Field

from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import LlmError, StageError
from frictiontune.core.json import JsonObject, dumps
from frictiontune.core.secrets import apply_redactions, context_window, detect_candidates
from frictiontune.domain.responses import (
    CompactedToolOutput,
    InferredToolSchema,
    RedactionReview,
)
from frictiontune.domain.trace import Message, ReasoningPart, TextPart, ToolPart
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from frictiontune.pipeline.context import StageContext

_COMPACT_PROMPT = "normalize_compaction"
_SCHEMA_PROMPT = "schema_inference"
_REDACT_PROMPT = "redaction_review"

# Candidates are reviewed in bounded batches so each reply stays small. The
# reviewer now names only the indices to redact (see ``RedactionReview``), so a
# batch reply is a handful of integers, not one object per candidate; the batch
# still bounds a single call's input and the conservative fallback's blast radius.
_REDACTION_BATCH = 64

# Tokens a single line number (or candidate index) costs in a reply — digits plus
# a separator, generously rounded up. Used to size a reply budget to the request:
# a chunk of N lines can name at most N lines, so its reply never needs more.
_TOKENS_PER_LINE = 6

# Schema inference reads the *shape* of a tool call, not its content, so examples
# are shown with long string values truncated and the example list capped. A tool
# whose calls carry large payloads (a file body, a big JSON argument) would
# otherwise build a prompt past a server slot's context, which the server rejects
# with a non-retryable 400 and — before this — aborted the whole stage.
_SCHEMA_VALUE_CHARS = 200
_SCHEMA_TOTAL_CHARS = 6_000


def _shrink_example(value: object, limit: int = _SCHEMA_VALUE_CHARS) -> object:
    """Truncate long strings (and cap list length) in a tool-call argument tree."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {key: _shrink_example(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_shrink_example(item, limit) for item in value[:10]]
    return value


def _schema_examples_text(examples: Sequence[JsonObject], total: int = _SCHEMA_TOTAL_CHARS) -> str:
    """Render tool-call examples for schema inference, bounded by ``total`` chars."""
    out: list[str] = []
    size = 0
    for example in examples:
        text = dumps(_shrink_example(example))
        if out and size + len(text) > total:
            break
        out.append(text)
        size += len(text) + 1
    return "\n".join(out)


# Structural keys never redacted: they identify the part, its role, and the tool
# linkage, so redacting one would corrupt the message rather than protect it.
_STRUCTURAL_KEYS = frozenset({"kind", "role", "name", "call_id", "status"})


def _iter_string_leaves(
    node: object, path: tuple[object, ...] = ()
) -> Iterator[tuple[tuple[object, ...], str]]:
    """Yield every string leaf in a JSON-like tree with its relative path.

    Structural keys are skipped only at the *top level* of a part, where they
    identify the part, its role, and the tool linkage. Inside a tool call's
    ``arguments`` a key named ``name`` or ``status`` is ordinary data and must
    still be scanned, so the skip must not be applied recursively.
    """
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            if not path and key in _STRUCTURAL_KEYS:
                continue
            yield from _iter_string_leaves(value, (*path, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_string_leaves(value, (*path, index))


def _get_at(node: Any, path: tuple[object, ...]) -> Any:
    """Return the value at ``path`` in a nested JSON-like structure."""
    for key in path:
        node = node[key]
    return node


def _set_at(node: Any, path: tuple[object, ...], value: Any) -> None:
    """Set the value at ``path`` in a nested JSON-like structure."""
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


class NormalizeSettings(StageSettings):
    """Knobs for compaction, schema inference, and redaction."""

    compact_outputs: bool = True
    # The size at which a tool output is worth an LLM call. Compaction is the
    # pipeline's dominant cost — one call per qualifying output, and 40% of
    # outputs clear the old 4 000-char default — so the threshold has to be
    # derived from the budget it serves rather than fixed globally. It exists to
    # keep a transcript inside the transcript-bearing stages'
    # `max_transcript_chars`; an output that occupies 8% of that budget cannot be
    # what breaks it, while one that occupies a quarter of it can. The `local`
    # overlay therefore sets a quarter of its 50 000-char budget (12 000), which
    # on the measured corpus cuts compaction calls by 63% (7 357 qualifying
    # outputs to 2 726 of 18 340). Raise it to trade transcript fidelity for
    # cost; lower it to trade the other way.
    compact_threshold_chars: int = 4_000
    # How much numbered text one compaction request may carry. A tool output is
    # numbered verbatim for the elision agent, so a 58 KB output produced a prompt
    # past a local server's per-slot context; the reply then truncated and the
    # stage aborted. Chunking bounds each request by construction while keeping
    # global line numbers, so a huge output is compacted in several calls whose
    # ranges are merged before a single elision pass.
    compact_chunk_chars: int = Field(default=24_000, ge=1)
    # A chunk is also bounded by line count, so the elision reply — which names at
    # most one line number per line shown — stays within the role's token cap even
    # when the lines are very short. See `_line_chunks`.
    compact_chunk_lines: int = Field(default=1_000, ge=1)
    # How much of the assistant's own text/reasoning to show the compaction agent
    # so it knows what each tool call was for. The LLM is cheap; context is not,
    # so this is generous.
    compact_context_chars: int = 4_000
    infer_schemas: bool = True
    schema_examples_per_tool: int = 12
    redact: bool = True
    batch_size: int = 64


class _Counters:
    """Mutable counters for one normalize pass."""

    def __init__(self) -> None:
        self.messages = 0
        self.compacted = 0
        self.compact_chunks = 0
        self.compact_failed = 0
        self.candidates = 0
        self.redacted = 0
        self.redact_failed = 0
        self.schema_failed = 0


def _intent_context(message: Message, limit: int) -> str:
    """Return the assistant's own text/reasoning around a tool call, for the agent.

    Eliding well needs to know *why* a tool was called, which the output alone
    does not reveal. The surrounding assistant text and reasoning carry that
    intent; tool outputs do not, so they are excluded (they are what is being
    compacted). The tail is kept because the plan closest to the call matters
    most, and the whole string is bounded so context stays cheap.
    """
    chunks = [
        part.text
        for part in message.parts
        if isinstance(part, (ReasoningPart, TextPart)) and part.text
    ]
    text = "\n".join(chunks).strip()
    return text[-limit:] if len(text) > limit else text


def _number_lines(text: str, *, start: int = 1) -> str:
    """Render ``text`` with 1-based line numbers for the elision agent.

    The agent points at line ranges; it never re-types the text. Numbering is
    plumbing, so the reference is exact rather than the model's guess at line
    positions. Splitting only on newline characters (not ``str.splitlines``)
    keeps the numbering aligned with :func:`elide_lines`, so no terminator is
    lost. ``start`` lets a *chunk* keep the global numbering of the whole output
    (see :func:`_line_chunks`), so the ranges it returns are absolute.
    """
    return "\n".join(f"{number:>5} | {line}" for number, line in enumerate(text.split("\n"), start))


def _line_chunks(
    lines: list[str], limit: int, max_lines: int = 0
) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(start_line, chunk)`` with ``start_line`` 1-based and global.

    A tool output is numbered verbatim for the elision agent, so a very long
    output produced a prompt larger than a local server's per-slot context; the
    server then had no room to generate and the reply came back truncated,
    aborting the stage. Grouping lines into chunks that each stay under ``limit``
    raw characters keeps every request inside a slot while the *global* line
    numbers still let the union of the chunks' ranges be applied in one pass.

    ``max_lines`` bounds a chunk by line count as well. The elision reply names at
    most one line number per line it was shown, so its budget scales with the
    chunk's line count; without a line bound a chunk of pathologically short
    lines could still exceed the role's token cap, which is exactly the runaway
    the budget is meant to cut off. A single line longer than ``limit`` becomes
    its own chunk rather than being split, so the model always sees whole lines
    and never invents an index.
    """
    start = 0
    size = 0
    for index, line in enumerate(lines):
        cost = len(line) + 1
        full = size + cost > limit
        long_enough = max_lines > 0 and index - start >= max_lines
        if index > start and (full or long_enough):
            yield start + 1, lines[start:index]
            start = index
            size = 0
        size += cost
    if start < len(lines):
        yield start + 1, lines[start:]


def elide_lines(text: str, lines: Iterable[int]) -> str:
    """Drop 1-based ``lines`` from ``text``, marking each gap.

    The agent decides *which* lines are non-decisive; every line it leaves alone
    is copied verbatim, so a failure it did not explicitly elide survives byte
    for byte. Line numbers outside the file are ignored, so a hallucinated number
    can only fail to elide, never corrupt the text or crash the stage. Splitting
    only on newline characters keeps the reconstruction byte-exact, including a
    trailing newline and any carriage returns.
    """
    text_lines = text.split("\n")
    total = len(text_lines)
    removed = bytearray(total)
    for line in lines:
        if 1 <= line <= total:
            removed[line - 1] = 1

    out: list[str] = []
    gap = 0

    def flush_gap() -> None:
        nonlocal gap
        if gap:
            noun = "line" if gap == 1 else "lines"
            out.append(f"… {gap} {noun} elided …")
            gap = 0

    for index, text_line in enumerate(text_lines):
        if removed[index]:
            gap += 1
            continue
        flush_gap()
        out.append(text_line)
    flush_gap()
    return "\n".join(out)


class Normalize(Stage):
    """Compact, infer schemas, and redact messages."""

    spec = StageSpec(
        name="normalize",
        tier=Tier.A,
        requires_llm=True,
        prompts=["normalize_compaction", "schema_inference", "redaction_review"],
        roles=[RoleName.COMPACTOR, RoleName.SCHEMA_INFERER],
        lead=Lead.CORE,
        roi="High",
        justification=(
            "Raw parts are not trainable, and secrets must not leave the host. "
            "Lossy head/tail truncation would cut errors out of the middle."
        ),
        evidence=["tool output p90 13 KB; 33.7% of calls exceed 4 KB"],
        cost=CostProfile(compute="mixed", api_usd="medium", maintenance="medium"),
        inputs={"messages": "jsonl"},
        outputs={
            "messages": "jsonl",
            "tool_schemas": "jsonl",
            "redactions": "json",
            "normalize_report": "json",
        },
        kill_criterion="Drop compaction only if every tool output fits the budget.",
    )
    settings_model = NormalizeSettings

    async def _compact(
        self,
        message: Message,
        ctx: StageContext,
        counters: _Counters,
        settings: NormalizeSettings,
    ) -> Message:
        role = ctx.config.role(RoleName.COMPACTOR)
        threshold = settings.compact_threshold_chars
        parts = list(message.parts)
        changed = False
        intent = _intent_context(message, settings.compact_context_chars)
        for index, part in enumerate(parts):
            if (
                isinstance(part, ToolPart)
                and part.output is not None
                and len(part.output) > threshold
            ):
                elided = await self._elide_lines(part, intent, role, ctx, counters, settings)
                if elided:
                    parts[index] = part.model_copy(
                        update={"output": elide_lines(part.output, elided)}
                    )
                    changed = True
                counters.compacted += 1
        return message.model_copy(update={"parts": parts}) if changed else message

    async def _elide_lines(
        self,
        part: ToolPart,
        intent: str,
        role: LlmRoleConfig,
        ctx: StageContext,
        counters: _Counters,
        settings: NormalizeSettings,
    ) -> list[int]:
        """Collect the elision line numbers for one tool output, chunk by chunk.

        The output is numbered with *global* line numbers and fed to the agent in
        chunks that each fit one server slot. A chunk whose reply cannot be
        obtained (truncation past the slot, a schema failure after repair) is not
        fatal: the agent's own contract says keeping a line is always correct, so
        that chunk contributes no numbers and its lines survive verbatim. One
        pathological output therefore degrades compaction, never the stage.
        """
        output = part.output or ""
        removed: list[int] = []
        for start, chunk in _line_chunks(
            output.split("\n"), settings.compact_chunk_chars, settings.compact_chunk_lines
        ):
            chunk_text = "\n".join(chunk)
            counters.compact_chunks += 1
            # The reply budget scales with the chunk: a chunk of N lines can name
            # at most N removable lines, so a global cap is both too small for a
            # big chunk's genuine elision and a target a runaway fills. The
            # observed failure enumerates line numbers *beyond* the chunk (the
            # local model marked 1,2,3,… to the cap); sizing the budget to the
            # chunk cuts that off at what the chunk could need, so the runaway
            # truncates and is rejected (the chunk is kept verbatim) while any
            # genuine elision — a subset of the chunk — still fits.
            call_role = role.model_copy(
                update={"max_tokens": min(role.max_tokens, len(chunk) * _TOKENS_PER_LINE + 256)}
            )
            try:
                result = await ctx.run.gateway.complete_structured(
                    RoleName.COMPACTOR,
                    call_role,
                    _COMPACT_PROMPT,
                    {
                        "tool_name": part.name,
                        "status": part.status or "unknown",
                        "exit_code": "non-zero" if part.is_error else "0",
                        "char_count": len(chunk_text),
                        "line_count": len(chunk),
                        "context": intent,
                        "output": _number_lines(chunk_text, start=start),
                    },
                    CompactedToolOutput,
                )
            except (LlmError, StageError):
                counters.compact_failed += 1
                continue
            removed.extend(result.elide)
        return removed

    async def _redact_message(
        self, message: Message, ctx: StageContext, counters: _Counters
    ) -> Message:
        """Detect and redact secrets across every string leaf of the message.

        Tool-call ``arguments`` are dumped verbatim into the trained assistant
        content (and into the corpus), so a fixed pair of top-level fields is not
        enough: every string leaf is scanned. A few structural keys (the part
        ``kind`` discriminator, role, tool name, call id, status) are skipped so
        redaction cannot break the message's structure or tool linkage.
        """
        data = message.model_dump(mode="json")
        found: list[tuple[int, tuple[object, ...], str, int, int, str]] = []
        for part_index, part in enumerate(data["parts"]):
            base: tuple[object, ...] = ("parts", part_index)
            for relative, value in _iter_string_leaves(part):
                for candidate in detect_candidates(value):
                    found.append(
                        (
                            len(found),
                            (*base, *relative),
                            value,
                            candidate.start,
                            candidate.end,
                            candidate.kind,
                        )
                    )
        if not found:
            return message

        role = ctx.config.role(RoleName.COMPACTOR)
        redact_indices: set[int] = set()
        counters.candidates += len(found)
        for offset in range(0, len(found), _REDACTION_BATCH):
            batch = found[offset : offset + _REDACTION_BATCH]
            lines = [
                f"{index} | {context_window(value, span_start, span_end)}"
                for index, _path, value, span_start, span_end, _kind in batch
            ]
            call_role = role.model_copy(
                update={"max_tokens": min(role.max_tokens, len(batch) * _TOKENS_PER_LINE + 128)}
            )
            try:
                review = await ctx.run.gateway.complete_structured(
                    RoleName.COMPACTOR,
                    call_role,
                    _REDACT_PROMPT,
                    {"candidates": "\n".join(lines)},
                    RedactionReview,
                )
            except (LlmError, StageError):
                # The reviewer could not be reached, or its reply did not
                # validate. Silence must not leak: redact the whole batch and keep
                # going, so one bad call degrades precision, never the stage and
                # never privacy.
                counters.redact_failed += 1
                redact_indices.update(index for index, *_ in batch)
                continue
            redact_indices.update(review.redact)

        by_path: dict[tuple[object, ...], list[tuple[int, int, str]]] = defaultdict(list)
        for global_index, path, _value, start, end, kind in found:
            if global_index in redact_indices:
                by_path[path].append((start, end, kind))
                counters.redacted += 1
        for path, spans in by_path.items():
            original = _get_at(data, path)
            _set_at(data, path, apply_redactions(original, spans))
        return Message.model_validate(data)

    async def _infer_schemas(
        self,
        observations: dict[str, list[JsonObject]],
        ctx: StageContext,
        settings: NormalizeSettings,
        counters: _Counters,
    ) -> list[InferredToolSchema]:
        if not settings.infer_schemas:
            return []
        role = ctx.config.role(RoleName.SCHEMA_INFERER)
        schemas: list[InferredToolSchema] = []
        for name in sorted(observations):
            examples = observations[name]
            if not examples:
                continue
            try:
                schema = await ctx.run.gateway.complete_structured(
                    RoleName.SCHEMA_INFERER,
                    role,
                    _SCHEMA_PROMPT,
                    {
                        "tool_name": name,
                        "count": len(examples),
                        "examples": _schema_examples_text(examples),
                    },
                    InferredToolSchema,
                )
            except (LlmError, StageError):
                # A tool whose schema cannot be inferred is skipped, not fatal:
                # the schemas are context for later stages, so one pathological
                # tool must not abort a stage that has already normalised every
                # message.
                counters.schema_failed += 1
                continue
            schemas.append(schema)
        return schemas

    async def run(self, ctx: StageContext) -> StageResult:
        """Stream messages, compacting/redacting and collecting tool schemas."""
        settings = ctx.settings(self.settings_model)
        counters = _Counters()
        observations: dict[str, list[JsonObject]] = defaultdict(list)
        sink = ctx.jsonl_sink("messages")

        async def process(message: Message) -> Message:
            counters.messages += 1
            current = message
            # Redact *before* compacting: compaction ships the whole tool output
            # to the endpoint, so running it first would leak any secret the
            # redaction pass is about to remove. Redaction only ever sends the
            # short candidate window to the reviewer, so doing it first is
            # strictly less exposure. The compaction agent then sees placeholders
            # instead of secret bytes, which it does not need to judge elision.
            if settings.redact:
                current = await self._redact_message(current, ctx, counters)
            if settings.compact_outputs:
                current = await self._compact(current, ctx, counters, settings)
            for part in current.parts:
                if isinstance(part, ToolPart) and part.arguments:
                    bucket = observations[part.name]
                    if len(bucket) < settings.schema_examples_per_tool:
                        bucket.append(part.arguments)
            return current

        batch: list[Message] = []

        async def flush() -> None:
            if not batch:
                return
            results = await asyncio.gather(*(process(message) for message in batch))
            for result in results:
                sink.write(result)
            batch.clear()

        for message in ctx.catalog.stream("messages", Message, batch_size=settings.batch_size):
            batch.append(message)
            if len(batch) >= settings.batch_size:
                await flush()
        await flush()
        sink.close()

        schemas = await self._infer_schemas(observations, ctx, settings, counters)
        ctx.emit_jsonl("tool_schemas", schemas)
        ctx.emit_json(
            "redactions",
            {
                "candidates": counters.candidates,
                "redacted": counters.redacted,
                "review_failed": counters.redact_failed,
                "tools_observed": len(observations),
            },
        )
        report = {
            "messages": counters.messages,
            "compacted": counters.compacted,
            "compact_chunks": counters.compact_chunks,
            "compact_failed": counters.compact_failed,
            "redaction_candidates": counters.candidates,
            "redacted_spans": counters.redacted,
            "redaction_reviews_failed": counters.redact_failed,
            "tools_observed": len(observations),
            "schemas_inferred": len(schemas),
            "schema_inference_failed": counters.schema_failed,
        }
        ctx.emit_json("normalize_report", report)
        ctx.emit_text(
            "report",
            "# Normalize\n\n"
            f"- messages: {counters.messages}\n"
            f"- tool outputs compacted: {counters.compacted} "
            f"({counters.compact_chunks} chunks, {counters.compact_failed} kept whole)\n"
            f"- redaction candidates: {counters.candidates} "
            f"(redacted {counters.redacted}, {counters.redact_failed} reviews "
            f"failed and were redacted wholesale)\n"
            f"- tool schemas inferred: {len(schemas)} "
            f"({counters.schema_failed} tools skipped)\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "messages": float(counters.messages),
                "compacted": float(counters.compacted),
                "compact_failed": float(counters.compact_failed),
                "redaction_candidates": float(counters.candidates),
                "redacted": float(counters.redacted),
                "redaction_reviews_failed": float(counters.redact_failed),
                "schemas": float(len(schemas)),
                "schema_inference_failed": float(counters.schema_failed),
            },
        )
