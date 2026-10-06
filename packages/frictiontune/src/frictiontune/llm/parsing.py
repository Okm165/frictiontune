"""Recovering a JSON object from model text.

Pure plumbing: tolerate prose and code fences, then return the first parsable
top-level object. No schema knowledge lives here.
"""

from __future__ import annotations

import json
from typing import Any


def _strip_code_fence(text: str) -> str:
    """Return the contents of the first fenced block, or the stripped text.

    A leading ``json``/``JSON`` language tag is dropped. Fence contents are
    preferred over the surrounding prose because a fence is an explicit "this is
    the payload" signal.
    """
    candidate = text.strip()
    fence_start = candidate.find("```")
    if fence_start == -1:
        return candidate
    fence_end = candidate.find("```", fence_start + 3)
    if fence_end == -1:
        return candidate
    inner = candidate[fence_start + 3 : fence_end].strip()
    first_newline = inner.find("\n")
    if first_newline != -1 and inner[:first_newline].strip().lower() in {"json", ""}:
        inner = inner[first_newline + 1 :]
    return inner.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first parsable top-level JSON object found in ``text``, else ``None``.

    Scanning resumes at every ``{`` instead of assuming the object spans the
    first brace to the last, so prose containing braces (``the set {x} …``)
    cannot swallow a later, valid object and trailing prose cannot invalidate an
    otherwise well-formed reply.
    """
    candidate = _strip_code_fence(text)
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError:
            start = candidate.find("{", start + 1)
            continue
        return parsed if isinstance(parsed, dict) else None
    return None
