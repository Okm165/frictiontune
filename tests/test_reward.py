"""Tests for the GRPO reward composition.

The reward must make a semantic call (the judge) and must not score a constant
for arbitrary text. A well-formed text answer makes no tool call, so the tool
term scores zero for it rather than granting free credit.
"""

from __future__ import annotations

import pytest
from frictiontune.core.errors import ConfigError
from frictiontune_train.reward import RewardComposer, _format_score, _tool_score


def test_format_score_penalises_empty_and_runaway() -> None:
    assert _format_score("", max_chars=100) == 0.0
    assert _format_score("hello", max_chars=100) == 1.0
    assert _format_score("x" * 101, max_chars=100) == 0.0


def test_tool_score_is_zero_without_a_tool_call() -> None:
    assert _tool_score("just a plain text answer") == 0.0


def test_tool_score_rewards_parsable_tool_calls() -> None:
    assert _tool_score('```json\n{"path": "a.py"}\n```') == 1.0
    assert _tool_score("```json\n{oops}\n```") == 0.0


def test_reward_composer_requires_a_judge_endpoint() -> None:
    with pytest.raises(ConfigError, match="judge"):
        RewardComposer()


def test_reward_without_a_judge_weight_never_calls_the_endpoint() -> None:
    # judge weight zero: the composer must not touch the network, so a bogus
    # URL is fine and the score is the deterministic terms only.
    composer = RewardComposer(
        weights={"judge": 0.0, "format": 1.0, "tool": 1.0},
        judge_url="http://127.0.0.1:0/never",
    )

    scores = composer(["plain text", '```json\n{"a": 1}\n```'])

    assert scores == [1.0, 2.0]
