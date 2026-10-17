"""Deterministic secret *detection* (never the redaction decision).

Detection is plumbing: it finds candidate spans with conservative regular
expressions. Whether a candidate is a real secret is a judgement made by the
``redaction_review`` agent. This split is deliberate — over-redaction destroys
evidence, under-redaction leaks data, and the model weighs that trade-off better
than a pattern.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence

_REDACTED = "[REDACTED:{kind}]"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    (
        "assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|authorization|bearer)"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-./+]{8,})[\"']?"
        ),
    ),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-./+]{16,})")),
    ("long_token", re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
)


class SecretCandidate(BaseModel):
    """A detected span that may or may not need redaction."""

    model_config = ConfigDict(extra="forbid")

    index: int
    start: int
    end: int
    kind: str


def _passes_luhn(value: str) -> bool:
    """Whether ``value``'s digits form a Luhn-valid 13-19 digit number.

    The cheap ``card`` pattern matches any 13-19 digit run, which in real
    transcripts is mostly order ids, timestamps, and hashes. Luhn is the
    industry checksum every real card number satisfies; applying it is pure
    validation, so the reviewer never sees the noise it removes.
    """
    digits = [int(ch) for ch in value if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def detect_candidates(text: str) -> list[SecretCandidate]:
    """Find non-overlapping candidate secret spans in ``text``."""
    spans: list[tuple[int, int, str]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start = match.start(1) if match.groups() else match.start(0)
            end = match.end(1) if match.groups() else match.end(0)
            if kind == "card" and not _passes_luhn(text[start:end]):
                continue
            spans.append((start, end, kind))
    spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    candidates: list[SecretCandidate] = []
    last_end = -1
    for start, end, kind in spans:
        if start < last_end:
            continue
        candidates.append(
            SecretCandidate(
                index=len(candidates),
                start=start,
                end=end,
                kind=kind,
            )
        )
        last_end = end
    return candidates


def context_window(text: str, start: int, end: int, *, radius: int = 60) -> str:
    """Return a short context string around a candidate span for the reviewer."""
    window_start = max(0, start - radius)
    window_end = min(len(text), end + radius)
    snippet = text[window_start:start] + " «" + text[start:end] + "» " + text[end:window_end]
    return snippet.replace("\n", " ")[:400]


def apply_redactions(text: str, spans: Sequence[tuple[int, int, str]]) -> str:
    """Replace ``spans`` with redaction markers, from last to first."""
    result = text
    for start, end, kind in sorted(spans, key=lambda span: span[0], reverse=True):
        result = result[:start] + _REDACTED.format(kind=kind) + result[end:]
    return result
