"""Tests for deterministic secret detection (never the redaction decision)."""

from __future__ import annotations

from frictiontune.core.secrets import (
    _passes_luhn,
    apply_redactions,
    detect_candidates,
)


def test_luhn_accepts_real_card_numbers() -> None:
    assert _passes_luhn("4242424242424242")
    assert _passes_luhn("4111 1111 1111 1111")
    assert _passes_luhn("378282246310005")  # 15-digit Amex


def test_luhn_rejects_arbitrary_digit_runs() -> None:
    assert not _passes_luhn("1234567890123456")
    assert not _passes_luhn("4895106293267")
    assert not _passes_luhn("4242424242424241")


def test_detection_drops_card_shaped_non_cards() -> None:
    # A 16-digit order id that fails Luhn must not become a candidate.
    assert [c.kind for c in detect_candidates("order 1234567890123456 shipped")] == []


def test_detection_keeps_a_luhn_valid_card() -> None:
    kinds = [c.kind for c in detect_candidates("pay 4242 4242 4242 4242 now")]

    assert kinds == ["card"]


def test_detection_finds_private_keys_and_assignments() -> None:
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"
    kinds = [c.kind for c in detect_candidates(f"{key}\napi_key = sk-abcdef123456")]

    assert "private_key" in kinds
    assert "assignment" in kinds


def test_candidates_are_non_overlapping_and_indexed() -> None:
    text = "a@b.com and c@d.com"
    candidates = detect_candidates(text)

    assert [c.index for c in candidates] == [0, 1]
    assert all(candidates[i].end <= candidates[i + 1].start for i in range(len(candidates) - 1))


def test_apply_redactions_marks_each_span() -> None:
    text = "token=abcdef123456 then done"
    candidates = detect_candidates(text)
    spans = [(c.start, c.end, c.kind) for c in candidates]

    redacted = apply_redactions(text, spans)

    assert "[REDACTED:" in redacted
    assert "abcdef123456" not in redacted
