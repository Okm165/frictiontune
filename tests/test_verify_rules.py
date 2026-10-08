"""Tests for the rule-verification gate's held-out sampling.

The verifier must not replay a rule on the episodes it was compiled from, or the
gate would accept a rule on its own evidence. These cover that selection without
an LLM endpoint.
"""

from __future__ import annotations

from frictiontune.domain.episodes import Episode
from frictiontune.stages.verify_rules import _held_out


def _episode(episode_id: str) -> Episode:
    """Build the minimal persisted episode needed to exercise selection."""
    return Episode(
        id=episode_id,
        session_id="s",
        start_user_message_id="m",
        start_seq=0,
        end_seq=1,
    )


def test_held_out_excludes_source_episodes() -> None:
    pool = [_episode("e1"), _episode("e2"), _episode("e3"), _episode("e4")]
    event_ids = {"e1": {"ev1"}, "e2": {"ev2"}, "e3": {"ev3"}, "e4": {"ev4"}}

    held_out = _held_out(pool, event_ids, ["ev1", "ev3"], 5)

    assert [episode.id for episode in held_out] == ["e2", "e4"]


def test_held_out_applies_limit_after_exclusion() -> None:
    pool = [_episode("e1"), _episode("e2"), _episode("e3")]
    event_ids = {"e1": {"ev1"}}

    held_out = _held_out(pool, event_ids, ["ev1"], 1)

    assert [episode.id for episode in held_out] == ["e2"]


def test_held_out_without_source_events_takes_first() -> None:
    pool = [_episode("e1"), _episode("e2")]

    held_out = _held_out(pool, {}, [], 1)

    assert [episode.id for episode in held_out] == ["e1"]


def test_held_out_returns_empty_when_all_episodes_are_source() -> None:
    pool = [_episode("e1"), _episode("e2")]
    event_ids = {"e1": {"ev1"}, "e2": {"ev1"}}

    held_out = _held_out(pool, event_ids, ["ev1"], 5)

    assert held_out == []
