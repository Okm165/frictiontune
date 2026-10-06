"""Tests for the replay dashboard and its direction-aware headline."""

from __future__ import annotations

import pytest
from frictiontune.reporting.dashboard import (
    PRIMARY_SIGNAL,
    build_dashboard,
    make_signal,
    relative_drop,
)


def test_lower_better_signal_improves_when_tuned_is_smaller() -> None:
    signal = make_signal("user_corrections", [3, 4, 5], [1, 2, 3], direction="lower_better")
    assert signal.improved
    assert signal.delta == pytest.approx(-2.0)


def test_higher_better_signal_improves_when_tuned_is_larger() -> None:
    signal = make_signal("friction_avoidance", [0.4, 0.5], [0.6, 0.7], direction="higher_better")
    assert signal.improved
    assert signal.delta == pytest.approx(0.2)


def test_signal_rejects_unpaired_lengths() -> None:
    with pytest.raises(ValueError):
        make_signal("x", [1, 2], [1], direction="lower_better")


def test_relative_drop_guards_zero_baseline() -> None:
    assert relative_drop(0.0, 5.0) == 0.0
    assert relative_drop(10.0, 8.0) == pytest.approx(0.2)


def test_headline_for_higher_better_primary() -> None:
    primary = make_signal(
        PRIMARY_SIGNAL, [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], direction="higher_better"
    )
    dashboard = build_dashboard([primary], episodes=3, seeds=[0, 1, 2])

    assert dashboard.friction_drop == pytest.approx(0.2)
    assert dashboard.friction_drop_ci_low == pytest.approx(0.2)
    assert dashboard.friction_drop_ci_high == pytest.approx(0.2)
    assert "friction drop" in dashboard.headline()


def test_headline_for_lower_better_primary_preserves_sign() -> None:
    primary = make_signal(PRIMARY_SIGNAL, [0.4, 0.4], [0.2, 0.2], direction="lower_better")
    dashboard = build_dashboard([primary], episodes=2, seeds=[0, 1])

    # tuned (0.2) < baseline (0.4) is an improvement, so the drop is positive.
    assert dashboard.friction_drop == pytest.approx(0.5)
    assert dashboard.friction_drop_ci_low == pytest.approx(0.5)
    assert dashboard.friction_drop_ci_high == pytest.approx(0.5)


def test_dashboard_without_primary_signal_is_zero() -> None:
    other = make_signal("tool_errors", [1, 2], [1, 1], direction="lower_better")
    dashboard = build_dashboard([other], episodes=2, seeds=[0])

    assert dashboard.friction_drop == 0.0
    assert dashboard.signal("tool_errors") is not None
