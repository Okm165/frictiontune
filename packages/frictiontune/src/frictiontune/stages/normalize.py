"""``normalize`` — turn raw parts into trainable, de-identified messages.

Three transforms, each split into plumbing plus judgment:

* **compaction** — long tool outputs are compressed by an agent that must keep
  error text verbatim (head/tail truncation would cut errors out of the middle);
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

from frictiontune.config.models import RoleName
from frictiontune.core.json import JsonObject, dumps
from frictiontune.core.secrets import apply_redactions, context_window, detect_candidates
from frictiontune.domain.responses import CompactedToolOutput, InferredToolSchema, RedactionReview
from frictiontune.domain.trace import Message, ToolPart
from frictiontune.pipeline.spec import CostProfile, Lead, StageSpec, Tier
from frictiontune.pipeline.stage import Stage, StageResult, StageSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from frictiontune.pipeline.context import StageContext

_COMPACT_PROMPT = "normalize_compaction"
_SCHEMA_PROMPT = "schema_inference"
_REDACT_PROMPT = "redaction_review"

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
    compact_threshold_chars: int = 4_000
    infer_schemas: bool = True
    schema_examples_per_tool: int = 12
    redact: bool = True
    batch_size: int = 64


class _Counters:
    """Mutable counters for one normalize pass."""

    def __init__(self) -> None:
        self.messages = 0
        self.compacted = 0
        self.candidates = 0
        self.redacted = 0


class Normalize(Stage):
    """Compact, infer schemas, and redact messages."""

    spec = StageSpec(
        name="normalize",
        tier=Tier.A,
        requires_llm=True,
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
        threshold: int,
    ) -> Message:
        role = ctx.config.role(RoleName.COMPACTOR)
        parts = list(message.parts)
        changed = False
        for index, part in enumerate(parts):
            if (
                isinstance(part, ToolPart)
                and part.output is not None
                and len(part.output) > threshold
            ):
                result = await ctx.run.gateway.complete_structured(
                    RoleName.COMPACTOR,
                    role,
                    _COMPACT_PROMPT,
                    {
                        "tool_name": part.name,
                        "status": part.status or "unknown",
                        "exit_code": "non-zero" if part.is_error else "0",
                        "char_count": len(part.output),
                        "output": part.output,
                    },
                    CompactedToolOutput,
                )
                parts[index] = part.model_copy(update={"output": result.kept})
                counters.compacted += 1
                changed = True
        return message.model_copy(update={"parts": parts}) if changed else message

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

        lines = [
            f"{global_index} | {context_window(value, start, end)}"
            for global_index, _path, value, start, end, _kind in found
        ]
        role = ctx.config.role(RoleName.COMPACTOR)
        review = await ctx.run.gateway.complete_structured(
            RoleName.COMPACTOR,
            role,
            _REDACT_PROMPT,
            {"candidates": "\n".join(lines)},
            RedactionReview,
        )
        decisions = {decision.index: decision for decision in review.decisions}

        by_path: dict[tuple[object, ...], list[tuple[int, int, str]]] = defaultdict(list)
        counters.candidates += len(found)
        for global_index, path, _value, start, end, kind in found:
            decision = decisions.get(global_index)
            redact = decision.redact if decision is not None else True
            if redact:
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
    ) -> list[InferredToolSchema]:
        if not settings.infer_schemas:
            return []
        role = ctx.config.role(RoleName.SCHEMA_INFERER)
        schemas: list[InferredToolSchema] = []
        for name in sorted(observations):
            examples = observations[name]
            if not examples:
                continue
            schema = await ctx.run.gateway.complete_structured(
                RoleName.SCHEMA_INFERER,
                role,
                _SCHEMA_PROMPT,
                {
                    "tool_name": name,
                    "count": len(examples),
                    "examples": "\n".join(dumps(example) for example in examples),
                },
                InferredToolSchema,
            )
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
            if settings.compact_outputs:
                current = await self._compact(
                    current, ctx, counters, settings.compact_threshold_chars
                )
            if settings.redact:
                current = await self._redact_message(current, ctx, counters)
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

        schemas = await self._infer_schemas(observations, ctx, settings)
        ctx.emit_jsonl("tool_schemas", schemas)
        ctx.emit_json(
            "redactions",
            {
                "candidates": counters.candidates,
                "redacted": counters.redacted,
                "tools_observed": len(observations),
            },
        )
        report = {
            "messages": counters.messages,
            "compacted": counters.compacted,
            "redaction_candidates": counters.candidates,
            "redacted_spans": counters.redacted,
            "tools_observed": len(observations),
            "schemas_inferred": len(schemas),
        }
        ctx.emit_json("normalize_report.json", report)
        ctx.emit_text(
            "report.md",
            "# Normalize\n\n"
            f"- messages: {counters.messages}\n"
            f"- tool outputs compacted: {counters.compacted}\n"
            f"- redaction candidates: {counters.candidates} "
            f"(redacted {counters.redacted})\n"
            f"- tool schemas inferred: {len(schemas)}\n",
        )
        return StageResult.from_context(
            ctx,
            metrics={
                "messages": float(counters.messages),
                "compacted": float(counters.compacted),
                "redaction_candidates": float(counters.candidates),
                "redacted": float(counters.redacted),
                "schemas": float(len(schemas)),
            },
        )
