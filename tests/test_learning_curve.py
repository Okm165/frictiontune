"""Tests for the learning-curve extension's deterministic plumbing.

The curve's sweep, subset draws, and report assembly are plumbing over numbers
a GPU run produces; the GPU-bound training itself is marked ``gpu`` and excluded
from the default suite.
"""

from __future__ import annotations

from frictiontune.domain.dataset import SFTRecord
from frictiontune.stages._extensions import (
    LearningCurve,
    _point_name,
    _subset,
    _sweep,
)


def _record(identifier: str) -> SFTRecord:
    return SFTRecord(id=identifier, source=identifier, track="recovery")


def test_sweep_deduplicates_sorts_and_expands_repeats() -> None:
    sweep = _sweep([1.0, 0.5, 0.25, 0.5], 2)
    assert sweep == [(0.25, 1), (0.25, 2), (0.5, 1), (0.5, 2), (1.0, 1), (1.0, 2)]


def test_sweep_clamps_fractions_into_range() -> None:
    assert [fraction for fraction, _ in _sweep([0.0, 2.0, 0.5], 1)] == [0.01, 0.5, 1.0]


def test_subset_is_deterministic_sorted_and_sized() -> None:
    records = [_record(f"r{index:02d}") for index in range(17)]
    first = _subset(records, 0.25, 1, seed=0)
    again = _subset(records, 0.25, 1, seed=0)
    other_seed = _subset(records, 0.25, 1, seed=1)

    assert len(first) == 4  # round(17 * 0.25)
    assert [record.id for record in first] == [record.id for record in again]
    assert [record.id for record in first] == sorted(record.id for record in first)
    assert {record.id for record in first} != {record.id for record in other_seed} or first[
        0
    ].id != other_seed[0].id


def test_subset_never_yields_zero_or_exceeds_the_pool() -> None:
    records = [_record(f"r{index}") for index in range(3)]
    assert len(_subset(records, 0.01, 1, seed=0)) == 1
    full = _subset(records, 1.0, 1, seed=0)
    assert len(full) == 3
    assert len({record.id for record in full}) == 3  # sampled without replacement


def test_point_names_are_filesystem_safe_and_ordered() -> None:
    assert _point_name(0.25, 1) == "curve_f025_r1"
    assert _point_name(1.0, 2) == "curve_f100_r2"


def test_curve_spec_is_gpu_gated_without_llm() -> None:
    spec = LearningCurve.spec
    assert spec.tier.value == "C"
    assert not spec.requires_llm
    assert "val" in spec.inputs
    assert spec.gated_by is not None
