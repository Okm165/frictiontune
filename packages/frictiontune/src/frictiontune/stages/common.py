"""Shared helpers for stage implementations.

Only plumbing lives here: transcript rendering, bounded concurrency, and
taxonomy validation. No stage re-implements these.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

from frictiontune.core.errors import StageError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from frictiontune.config.models import RoleName, RunConfig
    from frictiontune.domain.friction import FrictionEvent
    from frictiontune.domain.trace import Message

T = TypeVar("T")

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


def render_turns(messages: Sequence[Message], indices: Sequence[int]) -> str:
    """Render only the selected turn indices."""
    selected = [
        f"{index} | {messages[index].role} | "
        f"{clip(messages[index].content_text(), _TRANSCRIPT_QUOTE_LIMIT)}"
        for index in indices
        if 0 <= index < len(messages)
    ]
    return "\n".join(selected)


async def bounded_gather(
    factory: Callable[[T], Awaitable[object]],
    items: Sequence[T],
    *,
    limit: int,
) -> list[object]:
    """Run ``factory(item)`` for every item with at most ``limit`` in flight."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(item: T) -> object:
        async with semaphore:
            return await factory(item)

    return list(await asyncio.gather(*(_run(item) for item in items)))


def role_config(config: RunConfig, role: RoleName) -> object:
    """Return the role binding, raising a stage error if it is missing."""
    try:
        return config.role(role)
    except KeyError as exc:
        raise StageError("config", str(exc)) from exc


def validate_event(event: FrictionEvent, config: RunConfig) -> FrictionEvent:
    """Validate a friction event's axes against the configured taxonomy."""
    taxonomy = config.taxonomy
    errors: list[str] = []
    if event.form not in taxonomy.forms:
        errors.append(f"unknown form {event.form!r}")
    if event.cause not in taxonomy.causes:
        errors.append(f"unknown cause {event.cause!r}")
    if event.cost not in taxonomy.costs:
        errors.append(f"unknown cost {event.cost!r}")
    if event.resolution not in taxonomy.resolutions:
        errors.append(f"unknown resolution {event.resolution!r}")
    if event.leaf not in taxonomy.leaves:
        errors.append(f"unknown leaf {event.leaf!r}")
    else:
        form, cause = taxonomy.leaves[event.leaf]
        if event.form != form or event.cause != cause:
            errors.append(
                f"leaf {event.leaf!r} maps to form={form!r}/cause={cause!r}, "
                f"not {event.form!r}/{event.cause!r}"
            )
    if errors:
        raise StageError("friction_detect", f"invalid friction event: {'; '.join(errors)}")
    return event
