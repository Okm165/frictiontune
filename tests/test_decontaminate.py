"""Tests for the decontamination time split.

Generated (synthetic) examples now flow into the curated pool. They must always
train and never be held out, and they must not move the real-work time boundary.
Records are grouped by source episode, so an episode never straddles the split.
"""

from __future__ import annotations

from frictiontune.domain.dataset import SFTRecord
from frictiontune.stages.decontaminate import _time_split


def _record(identifier: str, source: str | None = None) -> SFTRecord:
    # By default each record is its own episode, mirroring one record per episode.
    return SFTRecord(id=identifier, source=source or identifier, track="recovery")


def test_generated_examples_train_and_do_not_move_the_boundary() -> None:
    real = [f"r{index}" for index in range(10)]
    synthetic = [f"g{index}" for index in range(80)]
    records = [_record(identifier) for identifier in [*real, *synthetic]]
    meta = {identifier: ("text", index) for index, identifier in enumerate(real)}
    meta.update(dict.fromkeys(synthetic, ("text", 0)))

    train, evaluation = _time_split(records, meta, set(synthetic), 0.8)

    # The boundary is over the 10 real records only: 2 held out, 8 trained.
    assert {record.id for record in evaluation} == {"r8", "r9"}
    assert {record.id for record in train} == {*real[:8], *synthetic}


def test_an_all_generated_pool_has_no_evaluation_split() -> None:
    records = [_record("g0"), _record("g1")]
    meta = {"g0": ("text", 0), "g1": ("text", 0)}

    train, evaluation = _time_split(records, meta, {"g0", "g1"}, 0.8)

    assert evaluation == []
    assert {record.id for record in train} == {"g0", "g1"}


def test_empty_pool_splits_into_two_empty_sides() -> None:
    train, evaluation = _time_split([], {}, set(), 0.8)

    assert train == []
    assert evaluation == []


def test_split_is_deterministic_for_equal_timestamps() -> None:
    records = [_record("b"), _record("a"), _record("c")]
    meta = {"a": ("text", 5), "b": ("text", 5), "c": ("text", 5)}

    train, evaluation = _time_split(records, meta, set(), 0.34)

    assert [record.id for record in train] == ["a"]
    assert [record.id for record in evaluation] == ["b", "c"]


def test_an_episode_is_never_split_across_the_boundary() -> None:
    # Episode "e1" yields two records (one per friction event) at the same time.
    # A record-level split would train on e1_a and evaluate on e1_b, leaking the
    # episode's context into training.
    records = [
        _record("e1_a", "e1"),
        _record("e1_b", "e1"),
        _record("x0"),
        _record("x1"),
        _record("x2"),
        _record("x3"),
    ]
    meta = {
        "e1_a": ("text", 1),
        "e1_b": ("text", 1),
        "x0": ("text", 2),
        "x1": ("text", 3),
        "x2": ("text", 4),
        "x3": ("text", 5),
    }

    train, evaluation = _time_split(records, meta, set(), 0.5)

    assert [record.id for record in train] == ["e1_a", "e1_b", "x0"]
    assert [record.id for record in evaluation] == ["x1", "x2", "x3"]
