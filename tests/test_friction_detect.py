"""Tests for friction-event ensemble merging and per-episode failure handling.

The ensemble only buys precision if a consensus event actually requires a
majority of passes. These lock the vote rule, the evidence-overlap clustering
that avoids double-counting the same friction, the recorded vote count, and the
rule that an episode whose ensemble could not hold a vote is *that episode's*
loss — counted, not fatal to the stage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from frictiontune.core.errors import StageError
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import (
    EvidenceSpan,
    FrictionAnalysis,
    FrictionEvent,
    FrictionEventDraft,
    FrictionProfile,
)
from frictiontune.stages.friction_detect import (
    _EpisodeOutcome,
    _healthy_passes,
    _merge_ensemble,
    _vote_threshold,
    absorb_batch,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _draft(leaf: str, turns: list[int], confidence: float) -> FrictionEventDraft:
    return FrictionEventDraft(
        cost="effort",
        resolution="user-correction",
        leaf=leaf,
        summary=f"{leaf} at {turns}",
        evidence=[EvidenceSpan(turn=turn, quote=f"quote {turn}") for turn in turns],
        confidence=confidence,
    )


def _turns(draft: FrictionEventDraft) -> list[int]:
    return [span.turn for span in draft.evidence]


def test_vote_threshold_is_a_strict_majority() -> None:
    assert _vote_threshold(2) == 2
    assert _vote_threshold(3) == 2
    assert _vote_threshold(4) == 3
    assert _vote_threshold(5) == 3


def test_healthy_passes_keeps_the_survivors_of_a_partial_ensemble() -> None:
    ok = FrictionAnalysis()

    healthy = _healthy_passes([ok, RuntimeError("boom"), ok], passes=3)

    assert healthy == [ok, ok]


def test_healthy_passes_raises_when_no_quorum_survives() -> None:
    # One survivor cannot reach the 2-of-3 threshold, so the failure surfaces
    # instead of the episode being silently dropped.
    with pytest.raises(RuntimeError, match="boom"):
        _healthy_passes([FrictionAnalysis(), RuntimeError("boom"), RuntimeError("boom")], passes=3)


def test_healthy_passes_keeps_everything_when_none_fail() -> None:
    first = FrictionAnalysis(abstain=True)
    second = FrictionAnalysis()

    assert _healthy_passes([first, second], passes=2) == [first, second]


def test_consensus_keeps_only_events_supported_by_a_majority() -> None:
    leaf = "ignored_constraint"
    passes = [
        # Both passes see the event at overlapping turns: consensus.
        [_draft(leaf, [3], 0.5)],
        # Only this pass sees the second event: no consensus, dropped.
        [_draft(leaf, [3, 4], 0.8), _draft("scope_overreach", [9], 0.9)],
    ]

    merged = _merge_ensemble(passes, threshold=_vote_threshold(2))

    assert len(merged) == 1
    draft, votes = merged[0]
    assert draft.leaf == leaf
    assert votes == 2
    # The representative is the highest-confidence member; confidence is the mean.
    assert _turns(draft) == [3, 4]
    assert draft.confidence == (0.5 + 0.8) / 2


def test_same_leaf_at_disjoint_turns_stays_distinct() -> None:
    leaf = "ignored_constraint"
    passes = [
        [_draft(leaf, [3], 0.6), _draft(leaf, [11], 0.7)],
        [_draft(leaf, [3], 0.9), _draft(leaf, [11], 0.4)],
    ]

    merged = _merge_ensemble(passes, threshold=_vote_threshold(2))

    # Two genuinely distinct occurrences, each backed by both passes.
    assert len(merged) == 2
    assert sorted(turn for draft, _ in merged for turn in _turns(draft)) == [3, 11]
    assert [votes for _draft_out, votes in merged] == [2, 2]


def test_non_overlapping_citations_do_not_form_a_consensus() -> None:
    passes = [
        [_draft("ignored_constraint", [3], 0.6)],
        [_draft("ignored_constraint", [7], 0.6)],
    ]

    assert _merge_ensemble(passes, threshold=_vote_threshold(2)) == []


def test_vote_count_reflects_supporting_passes_not_draft_count() -> None:
    leaf = "ignored_constraint"
    # Three passes, two support the event; threshold for 3 passes is 2.
    passes = [
        [_draft(leaf, [5], 0.4), _draft(leaf, [5], 0.9)],
        [_draft(leaf, [5], 0.6)],
        [],
    ]

    merged = _merge_ensemble(passes, threshold=_vote_threshold(3))

    assert len(merged) == 1
    _draft_out, votes = merged[0]
    assert votes == 2


def _event(episode_id: str, event_id: str) -> FrictionEvent:
    """Build one persisted event for an episode outcome."""
    return FrictionEvent(
        id=event_id,
        episode_id=episode_id,
        leaf="ignored_constraint",
        cost="effort",
        resolution="self-recovery",
        summary="s",
        evidence=[EvidenceSpan(turn=0, quote="q")],
        confidence=0.9,
        form="rule-following",
        cause="instruction-ambiguity",
    )


def _episode(episode_id: str) -> Episode:
    """Build the minimal persisted episode the batch accounting needs."""
    return Episode(
        id=episode_id,
        session_id="s1",
        start_user_message_id="m",
        start_seq=0,
        end_seq=1,
    )


def _absorb(
    batch: Sequence[Episode], results: Sequence[Any]
) -> tuple[FrictionProfile, list[Any], list[str], int]:
    """Absorb a batch into a fresh profile; return it with the writes and warnings."""
    profile = FrictionProfile()
    written: list[Any] = []
    warnings: list[str] = []
    logger = cast(
        "Any",
        SimpleNamespace(warning=lambda event, **fields: warnings.append(event)),
    )
    sink = cast("Any", SimpleNamespace(write=written.append))
    invalid = absorb_batch(batch, results, profile, sink, logger)
    return profile, written, warnings, invalid


def test_a_failed_episode_is_counted_not_fatal() -> None:
    batch = [_episode("e1"), _episode("e2")]
    results: list[Any] = [
        _EpisodeOutcome(episode_id="e1", events=[_event("e1", "ev1")]),
        StageError("friction_detect", "analyst ensemble degraded"),
    ]

    profile, written, warnings, _invalid = _absorb(batch, results)

    # The healthy episode still lands, and coverage is a statement about the
    # episodes that *were* analysed — the failure is visible, not averaged in.
    assert [event.id for event in written] == ["ev1"]
    assert profile.total_events == 1
    assert profile.episodes_analysed == 1
    assert profile.episodes_with_friction == 1
    assert profile.unanalysed == 1
    assert profile.coverage() == 1.0
    assert warnings == ["episode could not be analysed"]


def test_invalid_events_are_summed_across_a_batch() -> None:
    batch = [_episode("e1"), _episode("e2")]
    results: list[Any] = [
        _EpisodeOutcome(episode_id="e1", events=[_event("e1", "ev1")], invalid=2),
        _EpisodeOutcome(episode_id="e2", events=[], abstained=True, invalid=1),
    ]

    profile, _written, warnings, invalid = _absorb(batch, results)

    assert invalid == 3
    assert profile.abstentions == 1
    assert profile.unanalysed == 0
    assert warnings == []
