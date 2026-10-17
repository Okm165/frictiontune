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


def test_signal_is_significant_when_the_interval_excludes_zero() -> None:
    # Six pairs so the instrument is powered: below n>=6 `significant` is held
    # False by design, however clean the interval looks.
    signal = make_signal(
        "user_corrections",
        [3, 4, 5, 3, 4, 5],
        [1, 2, 3, 1, 2, 3],
        direction="lower_better",
    )

    assert signal.improved
    assert signal.powered
    assert signal.significant


def test_signal_is_not_significant_when_the_interval_crosses_zero() -> None:
    # Deltas of +5 and -5: the mean is zero and the interval spans it.
    signal = make_signal("noisy", [0.0, 10.0], [5.0, 5.0], direction="lower_better")

    assert not signal.significant


def test_signal_from_a_single_pair_is_never_significant() -> None:
    # One pair yields a degenerate interval [d, d]; it cannot support a claim.
    signal = make_signal("one", [1.0], [0.0], direction="lower_better")

    assert signal.improved
    assert not signal.significant


def test_sign_test_p_and_power_floor_are_reported_per_signal() -> None:
    # n=4: even all-agreeing deltas leave the sign-test floor at 0.125, so the
    # signal must refuse `significant` however clean the bootstrap interval is.
    signal = make_signal("x", [3, 3, 3, 3], [5, 5, 5, 5], direction="higher_better")
    assert signal.sign_test_p == pytest.approx(0.125)
    assert signal.min_attainable_p == pytest.approx(0.125)
    assert not signal.powered
    assert not signal.significant

    # n=6: the floor crosses 0.05 and a unanimous improvement becomes claimable.
    powered = make_signal("x", [3, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5], direction="higher_better")
    assert powered.min_attainable_p == pytest.approx(2**-5)
    assert powered.powered
    assert powered.significant


def test_sign_test_p_discards_ties() -> None:
    # One moving pair among four: p is the two-sided probability of 1 success
    # in 1 trial, not of 1 in 4 — ties carry no sign information.
    signal = make_signal("x", [3, 3, 3, 3], [3, 3, 3, 5], direction="higher_better")
    assert signal.sign_test_p == pytest.approx(1.0)


def test_dashboard_reports_instrument_power_and_headline_caveat() -> None:
    primary = make_signal(
        PRIMARY_SIGNAL,
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        direction="higher_better",
    )
    dashboard = build_dashboard([primary], episodes=4, seeds=[0])
    assert not dashboard.powered
    assert dashboard.min_attainable_p == pytest.approx(0.125)
    assert "underpowered" in dashboard.headline()
    assert "no effect claim is possible" in dashboard.headline()
