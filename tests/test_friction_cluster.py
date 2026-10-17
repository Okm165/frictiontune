"""Tests for behaviour-cluster coverage accounting.

The friction_cluster agent may name the same event under more than one cluster,
so coverage must count each known event once or it can exceed 100%.
"""

from __future__ import annotations

from frictiontune.domain.profiles import BehaviorDimension
from frictiontune.stages._extensions import _covered_events


def _dimension(name: str, *event_ids: str) -> BehaviorDimension:
    return BehaviorDimension(
        name=name,
        dimension=name,
        target_behavior=f"be {name}",
        event_ids=list(event_ids),
    )


def test_an_event_in_several_clusters_counts_once() -> None:
    clusters = [
        _dimension("a", "e1", "e2"),
        _dimension("b", "e2", "e3"),
        _dimension("c", "e2"),
    ]

    assert _covered_events(clusters, {"e1", "e2", "e3"}) == 3


def test_unknown_event_ids_are_not_covered() -> None:
    clusters = [_dimension("a", "e1", "hallucinated")]

    assert _covered_events(clusters, {"e1", "e2"}) == 1
