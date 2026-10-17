"""Shared helpers for stage implementations.

Only plumbing lives here: transcript rendering, bounded concurrency, and
taxonomy validation. No stage re-implements these.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

from frictiontune.core.errors import LlmError, StageError, ValidationError
from frictiontune.domain.friction import FrictionEvent
from frictiontune.domain.responses import InferredToolSchema

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from frictiontune.config.models import RunConfig
    from frictiontune.core.json import JsonObject
    from frictiontune.domain.trace import Message
    from frictiontune.pipeline.context import StageContext

T = TypeVar("T")
R = TypeVar("R")

_TRANSCRIPT_QUOTE_LIMIT = 1_500


def clip(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` characters with an explicit marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


def render_transcript(
    messages: Sequence[Message], *, quote_limit: int = _TRANSCRIPT_QUOTE_LIMIT
) -> str:
    """Render messages as ``turn_index | role | content`` lines for a prompt."""
    lines: list[str] = []
    for index, message in enumerate(messages):
        body = clip(message.content_text(), quote_limit)
        label: str = message.role
        if message.has_tool_error():
            label = f"{label}(tool-error)"
        lines.append(f"{index} | {label} | {body}")
    return "\n".join(lines)


def read_events_by_episode(ctx: StageContext) -> dict[str, list[FrictionEvent]]:
    """Group friction events by episode id (events are far smaller than episodes)."""
    grouped: dict[str, list[FrictionEvent]] = defaultdict(list)
    for event in ctx.catalog.stream("friction_events", FrictionEvent):
        grouped[event.episode_id].append(event)
    return grouped


def read_tool_schemas(ctx: StageContext) -> list[JsonObject]:
    """Load inferred tool schemas as plain JSON objects for a prompt."""
    if not ctx.catalog.has("tool_schemas"):
        return []
    return [
        schema.model_dump(mode="json")
        for schema in ctx.catalog.stream("tool_schemas", InferredToolSchema)
    ]


def analyst_prompt_values(
    config: RunConfig,
    *,
    session_id: str,
    episode_id: str,
    turn_count: int,
    tool_calls: int,
    tool_errors: int,
    compactions: int,
    transcript: str,
    max_chars: int,
) -> dict[str, object]:
    """Build the ``friction_analyst`` prompt values from episode metadata.

    The single source of truth for the analyst's variables, shared by the
    ``friction_detect`` stage and the golden-episode evaluation harness so the
    two cannot drift.
    """
    taxonomy = config.taxonomy
    return {
        "session_id": session_id,
        "episode_id": episode_id,
        "turn_count": turn_count,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "compactions": compactions,
        "transcript": clip(transcript, max_chars),
        "costs": ", ".join(taxonomy.costs),
        "resolutions": ", ".join(taxonomy.resolutions),
        "leaf_axes": "\n".join(
            f"- {leaf}: form={axis.form}, cause={axis.cause}"
            + (f" — {axis.description}" if axis.description else "")
            for leaf, axis in sorted(taxonomy.leaves.items())
        ),
    }


async def gather_tolerant(
    factory: Callable[[T], Awaitable[R]],
    items: Sequence[T],
) -> list[R | BaseException]:
    """Run ``factory(item)`` for every item, returning results and failures in order.

    A fan-out stage must treat one item's failure as *that item's* outcome, not
    the batch's: a single unrepairable LLM reply should not abort a stage that
    has already produced good work for its other items. Concurrency is not
    bounded here: every fan-out in this codebase calls an LLM, and the role
    semaphore in :class:`~frictiontune.llm.client.LlmClient` already bounds how
    many of those are in flight, so a second stage-level limit would only
    duplicate it.

    Each element is either a result or the :class:`BaseException` that item
    raised, in the same order as ``items``. The caller decides what a failure
    means for its domain — a skip with a recorded reason (``mine``, ``judge``,
    ``compile_rules``) or a conservative default (``verify_rules``) — so this
    helper stays pure plumbing and never invents a semantic judgment.
    """
    return list(await asyncio.gather(*(factory(item) for item in items), return_exceptions=True))


def failure_reason(exc: BaseException) -> str:
    """Map a per-item failure to a stable, report-facing skip reason.

    The labels distinguish the two failure modes an operator cares about when a
    fan-out stage reports skips: a reply that could not be validated into the
    schema even after repair, versus a call that failed at the transport or
    provider. Anything else is surfaced by its type name so nothing is hidden.
    """
    if isinstance(exc, ValidationError):
        return "schema_rejected"
    if isinstance(exc, LlmError):
        return "llm_failed"
    return f"error:{type(exc).__name__}"


def validate_event(event: FrictionEvent, config: RunConfig) -> FrictionEvent:
    """Validate a friction event's leaf, cost, and resolution against the taxonomy.

    ``form`` and ``cause`` are not checked for agreement with the leaf: they are
    *derived* from the leaf by the caller, so a mismatch cannot occur. What is
    still validated is that the leaf is known and that the cost and resolution
    are configured values.
    """
    taxonomy = config.taxonomy
    errors: list[str] = []
    if event.leaf not in taxonomy.leaves:
        errors.append(f"unknown leaf {event.leaf!r}")
    if event.cost not in taxonomy.costs:
        errors.append(f"unknown cost {event.cost!r}")
    if event.resolution not in taxonomy.resolutions:
        errors.append(f"unknown resolution {event.resolution!r}")
    if errors:
        raise StageError("friction_detect", f"invalid friction event: {'; '.join(errors)}")
    return event
