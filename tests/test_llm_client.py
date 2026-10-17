"""Tests for the LLM client's request shaping and transport retry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.cache import ContentCache
from frictiontune.core.retry import RetryPolicy
from frictiontune.llm.client import (
    LlmClient,
    _message_text,
    _reasoning_text,
    base_urls,
    build_response_format,
)
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


class _CapturingCompletions:
    """Records the kwargs of each ``create`` call and returns one fixed outcome."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcome = outcome

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._outcome


def _capturing_client(
    tmp_path: Path, completions: _CapturingCompletions
) -> tuple[LlmClient, LlmRoleConfig]:
    client = LlmClient(ContentCache(tmp_path))
    config = LlmRoleConfig(model="m", cache=False)
    endpoint = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._sdk[("", "EMPTY", config.timeout_s)] = cast("AsyncOpenAI", endpoint)
    return client, config


async def test_client_forwards_a_seed_to_the_provider(tmp_path: Path) -> None:
    completions = _CapturingCompletions(_completion())
    client, config = _capturing_client(tmp_path, completions)

    await client.complete(RoleName.TARGET, config, [ChatMessage.user("hi")], seed=7)

    assert completions.calls[0]["seed"] == 7


async def test_client_omits_the_seed_when_unset(tmp_path: Path) -> None:
    completions = _CapturingCompletions(_completion())
    client, config = _capturing_client(tmp_path, completions)

    await client.complete(RoleName.TARGET, config, [ChatMessage.user("hi")])

    assert "seed" not in completions.calls[0]


def test_seed_participates_in_the_cache_key(tmp_path: Path) -> None:
    client = LlmClient(ContentCache(tmp_path))
    config = LlmRoleConfig(model="m")

    def key(seed: int | None) -> str:
        return client._cache_key(
            RoleName.TARGET,
            config,
            [ChatMessage.user("hi")],
            None,
            None,
            force_json=False,
            schema_name="response",
            seed=seed,
        )

    assert key(1) != key(2)
    assert key(None) != key(0)


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


def test_reasoning_text_captures_the_provider_stream() -> None:
    assert _reasoning_text(SimpleNamespace(reasoning_content="step 1")) == "step 1"
    assert _reasoning_text(SimpleNamespace(reasoning_content=None)) == ""
    assert _reasoning_text(SimpleNamespace(content="answer")) == ""


async def test_client_forwards_extra_body_to_the_provider(tmp_path: Path) -> None:
    completions = _CapturingCompletions(_completion())
    client = LlmClient(ContentCache(tmp_path))
    config = LlmRoleConfig(
        model="m",
        cache=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    endpoint = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._sdk[("", "EMPTY", config.timeout_s)] = cast("AsyncOpenAI", endpoint)

    await client.complete(RoleName.COMPACTOR, config, [ChatMessage.user("hi")])

    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


async def test_client_omits_extra_body_when_empty(tmp_path: Path) -> None:
    completions = _CapturingCompletions(_completion())
    client, config = _capturing_client(tmp_path, completions)

    await client.complete(RoleName.TARGET, config, [ChatMessage.user("hi")])

    assert completions.calls[0]["extra_body"] is None


def test_extra_body_participates_in_the_cache_key(tmp_path: Path) -> None:
    client = LlmClient(ContentCache(tmp_path))

    def key(extra_body: dict[str, object]) -> str:
        config = LlmRoleConfig(model="m", extra_body=extra_body)
        return client._cache_key(
            RoleName.COMPACTOR,
            config,
            [ChatMessage.user("hi")],
            None,
            None,
            force_json=True,
            schema_name="response",
            seed=None,
        )

    assert key({}) != key({"chat_template_kwargs": {"enable_thinking": False}})


async def test_client_captures_reasoning_content(tmp_path: Path) -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"ok":true}', reasoning_content="because"),
                finish_reason="stop",
            )
        ],
        model="m",
        usage=None,
    )
    completions = _CapturingCompletions(completion)
    client, config = _capturing_client(tmp_path, completions)

    response = await client.complete(RoleName.FRICTION_ANALYST, config, [ChatMessage.user("hi")])

    assert response.text == '{"ok":true}'
    assert response.reasoning == "because"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ("",)),
        ("", ("",)),
        (" , ", ("",)),
        ("http://a:8000/v1", ("http://a:8000/v1",)),
        ("http://a:8000/v1/, http://b:8010/v1", ("http://a:8000/v1", "http://b:8010/v1")),
    ],
)
def test_base_urls_normalises_a_replica_list(raw: str | None, expected: tuple[str, ...]) -> None:
    assert base_urls(LlmRoleConfig(model="m", base_url=raw)) == expected


async def test_client_fans_out_across_replicas_round_robin(tmp_path: Path) -> None:
    client = LlmClient(ContentCache(tmp_path))
    config = LlmRoleConfig(model="m", base_url="http://a:8000/v1, http://b:8010/v1")

    first = client._endpoint(RoleName.TARGET, config)
    second = client._endpoint(RoleName.TARGET, config)
    third = client._endpoint(RoleName.TARGET, config)

    assert first is not second
    assert first is third
    assert set(client._sdk) == {
        ("http://a:8000/v1", "EMPTY", config.timeout_s),
        ("http://b:8010/v1", "EMPTY", config.timeout_s),
    }
    await client.aclose()


async def test_client_single_endpoint_is_unchanged(tmp_path: Path) -> None:
    client = LlmClient(ContentCache(tmp_path))
    config = LlmRoleConfig(model="m", base_url="http://a:8000/v1")

    assert client._endpoint(RoleName.TARGET, config) is client._endpoint(RoleName.TARGET, config)
    assert set(client._sdk) == {("http://a:8000/v1", "EMPTY", config.timeout_s)}
    await client.aclose()
