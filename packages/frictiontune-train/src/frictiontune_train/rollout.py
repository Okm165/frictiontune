"""Sandbox rollout via an external agent runner.

frictiontune deliberately does not reimplement an agent loop. Instead, rollout
speaks HTTP to whatever agent runner is already installed (OpenCode, Claude
Code, a custom ACP bridge, ...). The runner is configured by environment, so
nothing about a specific agent is baked into the repo:

* ``FRICTIONTUNE_ROLLOUT_URL`` — the runner endpoint (required).
* ``FRICTIONTUNE_ROLLOUT_TOKEN`` — optional bearer token.

The endpoint receives a :class:`RolloutRequest` and must return a
:class:`RolloutResult`-shaped JSON object.
"""

from __future__ import annotations

import os

from frictiontune.backends.models import RolloutRequest, RolloutResult, RolloutStep
from frictiontune.core.errors import BackendUnavailableError


class HttpRolloutBackend:
    """Runs rollouts against a configured HTTP agent runner."""

    def __init__(
        self, url: str | None = None, *, token: str | None = None, timeout_s: float = 600.0
    ) -> None:
        """Initialise the backend from explicit values or the environment."""
        self._url = url or os.environ.get("FRICTIONTUNE_ROLLOUT_URL")
        self._token = token or os.environ.get("FRICTIONTUNE_ROLLOUT_TOKEN")
        self._timeout = timeout_s

    async def rollout(self, request: RolloutRequest) -> RolloutResult:
        """Send one rollout request and parse the runner's response."""
        if not self._url:
            raise BackendUnavailableError(
                "rollout",
                "set FRICTIONTUNE_ROLLOUT_URL to an agent runner (ACP/HTTP) endpoint",
            )
        import httpx

        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._url,
                json=request.model_dump(mode="json"),
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        return self._parse(payload)

    def _parse(self, payload: object) -> RolloutResult:
        """Parse a runner payload into a typed :class:`RolloutResult`."""
        if not isinstance(payload, dict):
            raise BackendUnavailableError("rollout", "runner returned a non-object payload")
        raw_steps = payload.get("steps", [])
        steps: list[RolloutStep] = []
        if isinstance(raw_steps, list):
            for step in raw_steps:
                if isinstance(step, dict):
                    steps.append(RolloutStep.model_validate(step))
        return RolloutResult(
            steps=steps,
            finished=bool(payload.get("finished", False)),
            outcome=payload.get("outcome") if isinstance(payload.get("outcome"), str) else None,
        )
