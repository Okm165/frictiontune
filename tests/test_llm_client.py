"""Tests for the LLM client's request shaping and transport retry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.cache import ContentCache
from frictiontune.core.retry import RetryPolicy
from frictiontune.llm.client import LlmClient, _message_text, build_response_format
from frictiontune.llm.messages import ChatMessage
from frictiontune.llm.transport import TRANSPORT_RETRYABLE
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA: dict[str, object] = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def test_text_role_is_left_alone_without_force() -> None:
    config = LlmRoleConfig(model="m", response_format="text")

    assert build_response_format(config, schema=None, schema_name="x", force_json=False) is None


def test_force_json_upgrades_a_text_role() -> None:
    config = LlmRoleConfig(model="m", response_format="text")

    assert build_response_format(config, schema=None, schema_name="x", force_json=True) == {
        "type": "json_object"
    }


def test_json_object_role_requests_json() -> None:
    config = LlmRoleConfig(model="m", response_format="json_object")

    assert build_response_format(config, schema=None, schema_name="x", force_json=False) == {
        "type": "json_object"
    }


def test_json_schema_role_sends_the_schema() -> None:
    config = LlmRoleConfig(model="m", response_format="json_schema")

    built = build_response_format(config, schema=_SCHEMA, schema_name="Verdict", force_json=True)

    assert built == {
        "type": "json_schema",
        "json_schema": {"name": "Verdict", "schema": _SCHEMA},
    }


def test_json_schema_role_degrades_without_a_schema() -> None:
    config = LlmRoleConfig(model="m", response_format="json_schema")

    assert build_response_format(config, schema=None, schema_name="x", force_json=True) == {
        "type": "json_object"
    }
    # Even without `force_json`, a json_schema role still asks for JSON.
    assert build_response_format(config, schema=None, schema_name="x", force_json=False) == {
        "type": "json_object"
    }


@pytest.mark.parametrize(
    "exc",
    [
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
        httpx.TransportError,
    ],
)
def test_transport_retryable_includes_provider_transient_errors(
    exc: type[BaseException],
) -> None:
    assert issubclass(exc, TRANSPORT_RETRYABLE)


@pytest.mark.parametrize("exc", [BadRequestError, AuthenticationError, NotFoundError])
def test_transport_retryable_excludes_permanent_errors(exc: type[BaseException]) -> None:
    assert not issubclass(exc, TRANSPORT_RETRYABLE)


class _FakeCompletions:
    """Replays a queue of outcomes: an exception is raised, anything else returned."""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes

    async def create(self, **kwargs: object) -> object:
        del kwargs
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeEndpoint:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcomes))


def _completion() -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")
        ],
        model="m",
        usage=None,
    )


async def test_client_retries_a_provider_transient_error(tmp_path: Path) -> None:
    client = LlmClient(ContentCache(tmp_path))
    endpoint = _FakeEndpoint(
        [
            APIConnectionError(message="connection dropped", request=cast("Any", None)),
            _completion(),
        ]
    )
    config = LlmRoleConfig(
        model="m",
        cache=False,
        retry=RetryPolicy(max_attempts=3, initial_delay_s=0.0, jitter_s=0.0),
    )
    client._sdk[("", "EMPTY", config.timeout_s)] = cast("AsyncOpenAI", endpoint)

    response = await client.complete(RoleName.JUDGE, config, [ChatMessage.user("hi")])

    assert response.text == '{"ok":true}'


def test_message_text_concatenates_content_part_lists() -> None:
    message = SimpleNamespace(content=[{"type": "text", "text": "a"}, "b", {"text": "c"}])

    assert _message_text(message) == "a\nb\nc"


def test_message_text_degrades_unrecognised_content_to_empty() -> None:
    assert _message_text(SimpleNamespace(content=None)) == ""
    assert _message_text(SimpleNamespace(content={"unexpected": True})) == ""
