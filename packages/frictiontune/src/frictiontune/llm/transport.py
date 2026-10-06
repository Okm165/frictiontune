"""The provider transport-error taxonomy, shared by every LLM call.

:data:`frictiontune.core.retry.RETRYABLE` covers OS-level failures. The OpenAI
SDK does not raise those: it has its own hierarchy, and ``APIConnectionError``
(which ``APITimeoutError`` extends), ``RateLimitError``, and
``InternalServerError`` are not ``OSError`` subclasses. Without naming them, a
dropped connection or a 429 would never be retried.

This is the single definition of a retryable transport failure. Both the chat
client and the embedding backend pass it to the one retry primitive, so all LLM
calls retry the same errors by construction.
"""

from __future__ import annotations

from openai import APIConnectionError, InternalServerError, RateLimitError

from frictiontune.core.retry import RETRYABLE

TRANSPORT_RETRYABLE: tuple[type[BaseException], ...] = (
    *RETRYABLE,
    APIConnectionError,  # connection drops and timeouts (APITimeoutError extends this)
    RateLimitError,  # HTTP 429
    InternalServerError,  # HTTP 5xx
)
