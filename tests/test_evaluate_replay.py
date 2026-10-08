"""Tests for the replay stage's deterministic helpers.

These cover the format signal and the sampling-honesty notes without needing an
LLM endpoint or a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from frictiontune.config.loader import load_config
from frictiontune.config.models import RoleName
from frictiontune.core.errors import StageError
from frictiontune.stages.evaluate_replay import (
    EvaluateReplay,
    EvaluateReplaySettings,
    _format_valid,
    _replay_notes,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_replay_missing_role_becomes_a_stage_error(config_dir: Path) -> None:
    # A role that is not configured must surface as a clear StageError with the
    # adapter hint, not a raw config/KeyError from deep inside the stage.
    config = load_config(config_dir).model_copy(update={"llm": {}})
    ctx = cast("Any", SimpleNamespace(config=config))

    with pytest.raises(StageError, match="not configured"):
        EvaluateReplay()._role(ctx, RoleName.TARGET, EvaluateReplaySettings())
