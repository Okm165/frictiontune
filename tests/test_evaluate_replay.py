"""Tests for the replay stage's deterministic helpers.

These cover the format signal and the sampling-honesty notes without needing an
LLM endpoint or a GPU.
"""

from __future__ import annotations

from frictiontune.stages.evaluate_replay import (
    EvaluateReplaySettings,
    _format_valid,
    _replay_notes,
)


def test_format_valid_scores_empty_and_plain_output() -> None:
    assert _format_valid("   ") == 0.0
    # No JSON emitted means nothing malformed to penalise.
    assert _format_valid("just prose") == 1.0


def test_format_valid_measures_emitted_json() -> None:
    assert _format_valid('```json\n{"a": 1}\n```') == 1.0
    assert _format_valid('```json\n{"a": }\n```') == 0.0


def test_seeds_collapse_at_zero_temperature() -> None:
    notes = _replay_notes(EvaluateReplaySettings(sampling_temperature=0.0), [0, 1, 2], [0])

    assert any("collapsed" in note for note in notes)


def test_stochastic_sampling_is_reported() -> None:
    notes = _replay_notes(EvaluateReplaySettings(sampling_temperature=0.7), [0, 1, 2], [0, 1, 2])

    assert any("sampled 3" in note for note in notes)


def test_single_configured_seed_adds_no_sampling_note() -> None:
    notes = _replay_notes(EvaluateReplaySettings(sampling_temperature=0.0), [0], [0])

    assert not any("collapsed" in note for note in notes)
    assert not any("sampled" in note for note in notes)
