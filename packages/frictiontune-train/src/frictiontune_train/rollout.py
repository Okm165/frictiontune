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

from frictiontune.backends.models import RolloutRequest, RolloutResult
from frictiontune.core.errors import BackendUnavailableError
from pydantic import ValidationError


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
        from frictiontune.core.retry import RetryPolicy, with_retry
        from frictiontune.llm.transport import is_transport_retryable

        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        url = self._url
        async with httpx.AsyncClient(timeout=self._timeout) as client:

            async def _post() -> httpx.Response:
                response = await client.post(
                    url,
                    json=request.model_dump(mode="json"),
                    headers=headers,
                )
                response.raise_for_status()
                return response

            response = await with_retry(
                _post, policy=RetryPolicy(), retry_on=is_transport_retryable
            )
            try:
                payload = response.json()
            except ValueError as exc:
                raise BackendUnavailableError(
                    "rollout", "runner returned a non-JSON payload"
                ) from exc
        return self._parse(payload)

    def _parse(self, payload: object) -> RolloutResult:
        """Parse a runner payload into a strictly validated :class:`RolloutResult`.

        Only the contract's own keys are read, so a runner may add metadata;
        every contract field is then validated strictly, so a malformed agent
        payload fails loudly instead of being silently coerced or dropped.
        """
        if not isinstance(payload, dict):
            raise BackendUnavailableError("rollout", "runner returned a non-object payload")
        try:
            return RolloutResult.model_validate(
                {
                    "steps": payload.get("steps", []),
                    "finished": payload.get("finished", False),
                    "outcome": payload.get("outcome"),
                },
                strict=True,
            )
        except ValidationError as exc:
            raise BackendUnavailableError(
                "rollout", f"runner returned a malformed payload: {exc}"
            ) from exc
