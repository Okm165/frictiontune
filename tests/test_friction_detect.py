"""Tests for friction-event ensemble merging.

The ensemble only buys precision if a consensus event actually requires a
majority of passes. These lock the vote rule, the evidence-overlap clustering
that avoids double-counting the same friction, and the recorded vote count.
"""

from __future__ import annotations

from frictiontune.domain.friction import EvidenceSpan, FrictionEventDraft
from frictiontune.stages.friction_detect import (
    _merge_ensemble,
    _vote_threshold,
)


def _draft(leaf: str, turns: list[int], confidence: float) -> FrictionEventDraft:
    return FrictionEventDraft(
        form="rule-following",
        cause="instruction-ambiguity",
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
