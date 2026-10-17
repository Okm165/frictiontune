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
from frictiontune.llm.messages import ChatMessage
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


def _skip_ctx(report: dict[str, Any] | None) -> Any:
    """A stub context for the empty-holdout path, optionally with a report."""
    emitted: dict[str, Any] = {}

    def record(name: str, value: Any) -> None:
        emitted[name] = value

    ctx = SimpleNamespace(
        stage="evaluate_replay",
        run_id="test-run",
        outputs={},
        emit_json=record,
        emit_jsonl=record,
        emit_text=record,
        catalog=SimpleNamespace(
            has=lambda name: report is not None,
            read_json=lambda name: report,
        ),
    )
    return cast("Any", ctx), emitted


def test_empty_holdout_is_skipped_not_fatal() -> None:
    # A vacuous held-out set must not abort the stage: it emits an explicit,
    # well-formed empty dashboard so `export` — which needs only the adapter —
    # still completes (GAP-047).
    ctx, emitted = _skip_ctx({"total": 0, "dropped_contaminated": 0})

    result = EvaluateReplay()._skip_empty_holdout(ctx, [0])

    assert result.metrics == {"episodes": 0.0, "skipped": 1.0}
    dashboard = emitted["dashboard"]
    assert dashboard.episodes == 0
    assert dashboard.signals == []
    assert any("held-out set is empty" in note for note in dashboard.notes)
    assert "report" in emitted and "replay" in emitted


def test_empty_holdout_note_names_the_split_when_nothing_was_contaminated() -> None:
    # The first run with an empty holdout blamed contamination while the report
    # said `dropped_contaminated: 0` — the split itself had emptied it. The note
    # must not guess.
    ctx, emitted = _skip_ctx({"total": 0, "dropped_contaminated": 0})

    EvaluateReplay()._skip_empty_holdout(ctx, [0])

    note = emitted["dashboard"].notes[0]
    assert "time split put every curated record on the training side" in note
    assert "training leak" not in note


def test_empty_holdout_note_names_contamination_when_it_happened() -> None:
    ctx, emitted = _skip_ctx({"eval": 3, "dropped_contaminated": 3})

    EvaluateReplay()._skip_empty_holdout(ctx, [0])

    note = emitted["dashboard"].notes[0]
    assert "3 record(s) and all of them were then dropped as training leaks" in note


def _rules_ctx(rules: list[dict[str, Any]] | None) -> Any:
    """A stub catalog exposing (or not exposing) the rules artifact."""
    return cast(
        "Any",
        SimpleNamespace(
            catalog=SimpleNamespace(
                has=lambda name: rules is not None and name == "rules_verified",
                read_json=lambda name: {"generated_from": "compile_rules", "rules": rules},
            )
        ),
    )


def _rule(identifier: str, status: str = "active") -> dict[str, Any]:
    return {
        "id": identifier,
        "applicability": "when a claim rests on an external value",
        "instruction": f"verify at the source ({identifier})",
        "scope": {"task_tags": [], "tools": [], "paths": []},
        "verifier": {"tier": "semantic", "spec": "check the turn", "evidence": []},
        "rationale": "observed friction",
        "status": status,
    }


def test_active_rules_render_and_candidates_are_excluded() -> None:
    from frictiontune.stages.evaluate_replay import _load_active_rules

    ctx = _rules_ctx([_rule("r1"), _rule("r2", status="candidate")])
    text = _load_active_rules(ctx, EvaluateReplaySettings())

    assert "verify at the source (r1)" in text
    assert "r2" not in text
    assert text.startswith("## Runtime rules")


def test_inject_rules_without_the_artifact_fails_loudly() -> None:
    # A rules-free replay silently passing as a rules-injected one would be a
    # mislabeled measurement; the setting must fail without the artifact.
    from frictiontune.core.errors import StageError
    from frictiontune.stages.evaluate_replay import _load_active_rules

    with pytest.raises(StageError, match="no rules_verified artifact"):
        _load_active_rules(_rules_ctx(None), EvaluateReplaySettings())


def test_rules_go_in_front_within_the_same_budget() -> None:
    from frictiontune.stages.evaluate_replay import _messages_with_rules

    conversation = [
        ChatMessage(role="user", content="x" * 600),
        ChatMessage(role="assistant", content="y" * 600),
    ]
    rules = "R" * 500

    injected = _messages_with_rules(list(conversation), rules, max_context_chars=1_500)

    assert injected[0].role == "system" and injected[0].content == rules
    total = sum(len(message.content) for message in injected)
    assert total <= 1_500
    # The most recent turns survive; the budget came out of the oldest.
    assert injected[-1].content == "y" * 600


def test_context_budget_is_derived_from_the_training_window() -> None:
    from frictiontune.stages.evaluate_replay import _context_budget

    # Derived: the 8192-token window minus the generation reserve, at the
    # conservative 3 chars/token for code-heavy text.
    assert _context_budget(EvaluateReplaySettings(), 8192) == (8192 - 1024 - 256) * 3
    # An explicit value wins — a declared instrument change.
    assert _context_budget(EvaluateReplaySettings(max_context_chars=40_000), 8192) == 40_000
    # The derivation never collapses to zero on a tiny window.
    assert _context_budget(EvaluateReplaySettings(), 512) >= 1_000
