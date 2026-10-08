"""Tests for the simulated-user turn accounting.

The ``user_turns_per_episode`` signal must count the simulated user's turns, not
the model's replies; counting both would double a goal's apparent steering cost
and make the name a lie.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.domain.responses import SimulatedUserTurn
from frictiontune.llm.client import LlmResponse
from frictiontune.stages._extensions import EvaluateSimUser, EvaluateSimUserSettings


class _Gateway:
    """A stub gateway returning queued simulated-user turns."""

    def __init__(self, turns: list[SimulatedUserTurn]) -> None:
        self._turns = iter(turns)

    async def complete_structured(
        self,
        role: object,
        config: object,
        prompt: object,
        values: object,
        schema: object,
    ) -> SimulatedUserTurn:
        return next(self._turns)

    async def complete_text(self, role: object, config: object, messages: object) -> LlmResponse:
        return LlmResponse(text="ok", model="stub")


def _run(turns: list[SimulatedUserTurn], *, max_turns: int) -> int:
    config = LlmRoleConfig(model="stub")
    ctx = SimpleNamespace(
        config=SimpleNamespace(role=lambda name: config),
        run=SimpleNamespace(gateway=_Gateway(turns)),
    )
    return asyncio.run(
        EvaluateSimUser()._run_scenario(
            "goal",
            RoleName.TARGET,
            config,
            cast("Any", ctx),
            EvaluateSimUserSettings(max_turns=max_turns),
        )
    )


def test_sim_user_counts_only_user_turns_until_resolved() -> None:
    turns = [
        SimulatedUserTurn(message="try again", resolved=False),
        SimulatedUserTurn(message="done", resolved=True),
    ]

    # Two user turns, with one assistant reply in between; the reply is not counted.
    assert _run(turns, max_turns=5) == 2


def test_sim_user_counts_max_turns_when_never_resolved() -> None:
    turns = [SimulatedUserTurn(message=str(index)) for index in range(3)]

    assert _run(turns, max_turns=3) == 3
