"""Tests for the golden-episode evaluation harness (offline plumbing only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from frictiontune.config.loader import load_config
from frictiontune.evaluation import (
    EpisodeOutcome,
    GoldenReport,
    load_bundled_golden,
    load_golden,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_bundled_fixture_loads() -> None:
    episodes = load_bundled_golden()

    assert len(episodes) >= 7
    assert len({episode.id for episode in episodes}) == len(episodes)
    assert all(episode.transcript.strip() for episode in episodes)


def test_bundled_fixture_labels_match_the_taxonomy(config_dir: Path) -> None:
    taxonomy = load_config(config_dir).taxonomy

    for episode in load_bundled_golden():
        for expected in episode.expected:
            assert expected.leaf in taxonomy.leaves, f"{episode.id}: {expected.leaf}"
            form, cause = taxonomy.leaves[expected.leaf]
            assert (expected.form, expected.cause) == (form, cause), episode.id


def test_load_golden_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '# a comment\n\n{"id": "a", "transcript": "x", "expected": []}\n',
        encoding="utf-8",
    )

    assert [episode.id for episode in load_golden(path)] == ["a"]


def test_report_scores_precision_recall_and_axes() -> None:
    report = GoldenReport(
        [
            EpisodeOutcome("e1", predicted={"low": ("f1", "c1")}, expected={"low": ("f1", "c1")}),
            EpisodeOutcome("e2", predicted={"a": ("f", "c")}, expected={"b": ("f", "c")}),
            EpisodeOutcome("e3", predicted={}, expected={"x": ("f", "c")}, abstained=True),
        ]
    )

    assert report.true_positive == 1
    assert report.false_positive == 1
    assert report.false_negative == 2
    assert report.axis_correct == 1
    assert report.exact_matches == 1
    assert report.abstentions == 1
    assert report.precision() == pytest.approx(0.5)
    assert report.recall() == pytest.approx(1 / 3)
    assert "golden episode" in report.summary()


def test_report_records_errors_without_scoring_them() -> None:
    report = GoldenReport([EpisodeOutcome("e1", error="boom")])

    assert report.errors == 1
    assert report.scored == []
    assert "boom" in report.summary()
