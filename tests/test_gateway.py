"""Tests for strict schema validation inside the single LLM gateway.

The gateway may see more than one top-level object in a reply (an example
followed by the answer, or a stray object). It must select the candidate that
validates and feed back a concise problem when none does.
"""

from __future__ import annotations

from frictiontune.domain.trace import DomainModel
from frictiontune.domain.types import NonNegativeInt
from frictiontune.llm.gateway import LlmGateway


class _Verdict(DomainModel):
    ok: bool
    n: NonNegativeInt = 0


def test_validate_picks_the_candidate_that_validates() -> None:
    text = 'Example: {"ok": "not-a-bool"} Answer: {"ok": true, "n": 3}'

    assert LlmGateway._validate(text, _Verdict) == _Verdict(ok=True, n=3)


def test_validate_reports_a_concise_problem_without_echoing_input() -> None:
    problem = LlmGateway._validate('{"ok": "yes"}', _Verdict)

    assert isinstance(problem, str)
    assert "ok" in problem
    assert "yes" not in problem


def test_validate_reports_a_missing_object() -> None:
    assert LlmGateway._validate("no json here", _Verdict) == "reply contained no JSON object"


def test_validate_accepts_an_integral_float_for_an_int_field() -> None:
    assert LlmGateway._validate('{"ok": true, "n": 3.0}', _Verdict) == _Verdict(ok=True, n=3)
