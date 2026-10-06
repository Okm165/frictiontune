"""Recovering a JSON object from model text.

Pure plumbing: tolerate prose and code fences, then return the first parsable
top-level object. No schema knowledge lives here.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first top-level JSON object found in ``text``, else ``None``."""
    candidate = text.strip()

    fence_start = candidate.find("```")
    if fence_start != -1:
        fence_end = candidate.find("```", fence_start + 3)
        if fence_end != -1:
            inner = candidate[fence_start + 3 : fence_end].strip()
            first_newline = inner.find("\n")
            if first_newline != -1 and inner[:first_newline].strip().lower() in {"json", ""}:
                inner = inner[first_newline + 1 :]
            candidate = inner.strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
