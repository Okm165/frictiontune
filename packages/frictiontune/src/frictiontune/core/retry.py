"""Centralised retry policy.

Every network call (LLM, embeddings, downloads) goes through :func:`with_retry`.
There is exactly one retry implementation in the codebase; stages never roll
their own loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")

# Transient OS-level transport failures. Provider SDKs raise their *own* error
# types, which are not OSError subclasses; those extend this set at the transport
# layer (see ``frictiontune.llm.transport.TRANSPORT_RETRYABLE``). Domain and
# validation errors are handled elsewhere.
RETRYABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class RetryPolicy(BaseModel):
    """Declarative retry configuration shared by all callers."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=4, ge=1)
    initial_delay_s: float = Field(default=0.5, ge=0.0)
    max_delay_s: float = Field(default=30.0, ge=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter_s: float = Field(default=0.25, ge=0.0)

    def wait(self) -> wait_exponential_jitter:
        """Build the tenacity wait strategy for this policy."""
        return wait_exponential_jitter(
            initial=self.initial_delay_s,
            max=self.max_delay_s,
            exp_base=self.multiplier,
            jitter=self.jitter_s,
        )

    @classmethod
    def immediate(cls, attempts: int) -> RetryPolicy:
        """A policy that retries at once, for cheap in-process work.

        Schema repair re-asks a model that has just failed validation; a backoff
        would only add latency, so every retry is immediate.
        """
        return cls(
            max_attempts=attempts,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
        )


def _log_retry(*, logger: object | None, retry_state: RetryCallState) -> None:
    if logger is None:
        return
    log = getattr(logger, "warning", None)
    if callable(log):
        log(
            "retrying call",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        )


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE,
    logger: object | None = None,
) -> T:
    """Await ``operation`` with the shared retry policy.

    ``operation`` must be a zero-argument coroutine factory so each attempt
    starts fresh (important for HTTP clients and generators).
    """
    active = policy or RetryPolicy()
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(active.max_attempts),
        wait=active.wait(),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
        before_sleep=lambda state: _log_retry(logger=logger, retry_state=state),
    ):
        with attempt:
            return await operation()
    raise AssertionError("unreachable: tenacity always exits via return or raise")
