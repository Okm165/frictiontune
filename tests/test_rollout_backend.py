"""Tests for the HTTP rollout backend's strict runner-payload parsing."""

from __future__ import annotations

import pytest
from frictiontune.core.errors import BackendUnavailableError
from frictiontune_train.rollout import HttpRolloutBackend


def test_parse_accepts_a_well_formed_payload() -> None:
    result = HttpRolloutBackend()._parse(
        {
            "steps": [
                {
                    "turn": 0,
                    "assistant_text": "hi",
                    "tool_name": "bash",
                    "tool_args": {"cmd": "ls"},
                    "tool_result": "ok",
                    "tool_error": False,
                }
            ],
            "finished": True,
            "outcome": "done",
        }
    )

    assert result.finished is True
    assert result.outcome == "done"
    assert result.steps[0].tool_name == "bash"


def test_parse_ignores_extra_runner_metadata() -> None:
    result = HttpRolloutBackend()._parse({"steps": [], "finished": False, "trace_id": "abc"})

    assert result.finished is False
    assert result.steps == []


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"steps": [], "finished": "false"},  # a string bool must not coerce to True
        {"steps": [{"turn": "0", "assistant_text": "hi"}], "finished": True},
        {"steps": [{"assistant_text": "hi"}]},  # missing required `turn`
    ],
)
def test_parse_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(BackendUnavailableError):
        HttpRolloutBackend()._parse(payload)
