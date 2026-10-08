"""Recovering a JSON object from model text.

Pure plumbing: tolerate prose and code fences, then return parsable *top-level*
objects. No schema knowledge lives here. Braces inside JSON strings are tracked,
so a valid object whose string values contain braces or backtick fences is
recovered intact, and a truncated object cannot leak an object embedded in one
of its own strings.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_QUOTE = '"'
_BACKSLASH = "\\"


def _top_level_object_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` spans of balanced top-level ``{...}`` objects.

    The scanner tracks string and escape state, so braces that appear inside a
    JSON string are never mistaken for structure. A truncated object never
    closes at depth zero, so it yields no span at all rather than exposing an
    inner object hidden in one of its strings.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == _BACKSLASH:
                escaped = True
            elif char == _QUOTE:
                in_string = False
            continue
        if char == _QUOTE:
            if depth > 0:
                in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield start, index + 1
                    start = -1


def iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield every parsable top-level JSON object in ``text``, in order.

    A whole-text fast path handles the common ``response_format`` case of a reply
    that is exactly one object. Otherwise balanced spans are decoded in order;
    spans that are not valid JSON (a prose brace, a broken example) are skipped.
    Callers that know the intended schema can select the candidate that
    validates rather than trusting the first braces in the reply.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            whole = json.loads(stripped)
        except json.JSONDecodeError:
            whole = None
        if isinstance(whole, dict):
            yield whole
            return
    for start, end in _top_level_object_spans(text):
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first parsable top-level JSON object found in ``text``, else ``None``.

    Scanning respects JSON string boundaries, so prose containing braces
    (``the set {x} …``) cannot swallow a later, valid object, trailing prose
    cannot invalidate an otherwise well-formed reply, and a fence or brace inside
    a string value cannot truncate the real payload.
    """
    return next(iter_json_objects(text), None)
