"""Tests for the decontamination time split.

Generated (synthetic) examples now flow into the curated pool. They must always
train and never be held out, and they must not move the real-work time boundary.
Records are grouped by source session, so a session never straddles the split.
"""

from __future__ import annotations

from frictiontune.domain.dataset import SFTRecord
from frictiontune.stages.decontaminate import (
    _contamination_report,
    _time_split,
    _val_split,
)


def _record(identifier: str, source: str | None = None, session: str | None = None) -> SFTRecord:
    # By default each record is its own episode, mirroring one record per episode.
    meta = {"session_id": session} if session else {}
    return SFTRecord(id=identifier, source=source or identifier, track="recovery", meta=meta)


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


def test_a_session_is_never_split_across_the_boundary() -> None:
    # Session "ses1" yields two records from *different episodes* at the same
    # time. A record- or episode-level split would train on one and evaluate on
    # the other, leaking the session's task and context into training.
    records = [
        _record("e1_a", "ses1:msg1", session="ses1"),
        _record("e1_b", "ses1:msg2", session="ses1"),
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


def test_report_flags_an_empty_eval_set_when_training_records_exist() -> None:
    report, body = _contamination_report(
        total=5,
        val=0,
        generated_train=0,
        train=5,
        surviving=0,
        dropped=5,
        quantile=0.8,
        reasons={"x": "t3/t4-semantic"},
    )

    assert report["eval_empty"] is True
    assert report["eval"] == 0
    assert "WARNING" in body


def test_report_does_not_flag_a_populated_eval_set() -> None:
    report, body = _contamination_report(
        total=5,
        val=0,
        generated_train=0,
        train=4,
        surviving=1,
        dropped=0,
        quantile=0.8,
        reasons={},
    )

    assert report["eval_empty"] is False
    assert "WARNING" not in body


def test_report_does_not_flag_an_all_generated_pool() -> None:
    # No real training records means there is nothing to hold out, so an empty
    # eval side is expected and must not read as a failure.
    report, _body = _contamination_report(
        total=2,
        generated_train=2,
        train=2,
        val=0,
        surviving=0,
        dropped=0,
        quantile=0.8,
        reasons={},
    )

    assert report["eval_empty"] is False


def test_a_large_newest_session_does_not_swallow_the_held_out_share() -> None:
    # The shape that produced an empty eval set on the real corpus: 32 records,
    # 18 sessions, and 9 of them in one newest session. Walking the groups
    # oldest-first and stopping at the boundary sent the straddling group wholly
    # to *train* (23 accumulated + 9 = 32), so nothing was held out and the
    # replay had no number to report.
    records = [_record(f"old{index}", session=f"ses_old{index}") for index in range(23)]
    records += [_record(f"new{index}", session="ses_new") for index in range(9)]
    meta = {f"old{index}": ("text", index) for index in range(23)}
    meta.update({f"new{index}": ("text", 100) for index in range(9)})

    train, evaluation = _time_split(records, meta, set(), 0.8)

    assert {record.id for record in evaluation} == {f"new{index}" for index in range(9)}
    assert len(train) == 23


def test_a_single_session_pool_still_holds_something_out() -> None:
    # One session means one group, and the group is bigger than the held-out
    # share. It must still be held out: an empty eval set is never the better
    # answer when there is real work to measure on.
    records = [_record(f"r{index}", session="ses1") for index in range(5)]
    meta = {f"r{index}": ("text", index) for index in range(5)}

    train, evaluation = _time_split(records, meta, set(), 0.8)

    assert len(evaluation) == 5
    assert train == []


def test_val_split_carves_newest_sessions_and_keeps_train_whole() -> None:
    # Five sessions of two records each. 20% of 10 records = 2, so the newest
    # session goes to val and the rest stay in train.
    records = [
        _record(f"s{session}_{part}", session=f"s{session}")
        for session in range(5)
        for part in range(2)
    ]
    meta = {record.id: ("text", int(record.id[1])) for record in records}

    train, val = _val_split(records, meta, 0.2)

    assert {record.id for record in val} == {"s4_0", "s4_1"}
    assert {record.id for record in train} == {
        f"s{session}_{part}" for session in range(4) for part in range(2)
    }


def test_val_split_never_takes_the_last_session() -> None:
    # One session holding everything: a val carve would starve training, so the
    # val set must come out empty rather than strip train to nothing.
    records = [_record(f"a{i}", session="ses_a") for i in range(6)]
    meta = {record.id: ("text", i) for i, record in enumerate(records)}

    train, val = _val_split(records, meta, 0.2)

    assert val == []
    assert len(train) == 6


def test_val_split_of_an_empty_pool_is_empty() -> None:
    assert _val_split([], {}, 0.2) == ([], [])


def test_track_counts_make_the_synthetic_share_visible() -> None:
    from frictiontune.stages.decontaminate import _track_counts

    records = [
        _record("a", session="s1"),
        _record("b", session="s1"),
        SFTRecord(id="c", source="c", track="success"),
        SFTRecord(id="d", source="d", track="synthetic"),
    ]
    assert _track_counts(records) == {"recovery": 2, "success": 1, "synthetic": 1}
    assert _track_counts([]) == {}
