"""Guards for strict agent-response validation and the single repair retry.

Agent responses are validated with ``strict=True``, so a stringified number or
boolean is a malformed reply that the gateway repairs rather than coerces. These
tests pin that behaviour and keep every response schema strict-compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from frictiontune.config.models import LlmRoleConfig, RoleName
from frictiontune.core.errors import ValidationError as FtValidationError
from frictiontune.domain import (
    Attribution,
    CompactedToolOutput,
    CompiledRules,
    ContaminationVerdict,
    DedupVerdict,
    EpisodePhases,
    FrictionAnalysis,
    FrictionClusters,
    InferredToolSchema,
    JudgeVerdict,
    RecoverySpan,
    RedactionReview,
    ReduceSummary,
    RouterDecision,
    RuleVerdict,
    SimulatedUserTurn,
    SyntheticBatch,
    VerifierTier,
)
from frictiontune.llm.client import LlmClient, LlmResponse
from frictiontune.llm.gateway import LlmGateway
from frictiontune.llm.prompts.registry import default_registry
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.llm.messages import ChatMessage, ToolSpec

# One representative, JSON-shaped reply per agent-response schema. Every entry
# must validate under strict mode; adding a schema with a stringified-scalar or
# enum field will fail here rather than at run time.
_SAMPLES: dict[type[BaseModel], dict[str, object]] = {
    CompactedToolOutput: {
        "kept": "code",
        "preserved_failure": True,
        "elided_lines": 3,
        "rationale": "kept the traceback",
    },
    InferredToolSchema: {"name": "read", "parameters": {"path": {"type": "string"}}},
    RedactionReview: {"decisions": [{"index": 0, "redact": False, "reason": "public"}]},
    JudgeVerdict: {
        "reason": "solid",
        "task_adherence": 5,
        "correctness": 5,
        "friction_avoidance": 5,
        "clarity": 5,
        "evidence": 5,
        "accept": True,
    },
    DedupVerdict: {"reason": "same lesson", "duplicate": True, "similarity": 1, "keep": "a"},
    RuleVerdict: {
        "rationale": "fires and prevents",
        "fires": True,
        "prevents": True,
        "collateral": False,
        "confidence": 1,
        "evidence_turns": [1, 2],
    },
    FrictionClusters: {
        "clusters": [
            {
                "name": "verify-before-claiming",
                "dimension": "verification",
                "target_behavior": "run the tests before reporting success",
                "event_ids": ["e1"],
                "rationale": "shared theme",
            }
        ]
    },
    SyntheticBatch: {
        "examples": [{"context": "ctx", "completion": "ok", "rationale": "teaches X"}]
    },
    SimulatedUserTurn: {"message": "please continue", "resolved": False, "reason": "open"},
    ContaminationVerdict: {"reason": "same task", "contaminated": True, "tier": "t4_task"},
    ReduceSummary: {
        "summary": "0 | user | hi",
        "preserved_failures": 1,
        "dropped_turns": 2,
        "rationale": "kept the error",
    },
    FrictionAnalysis: {
        "events": [
            {
                "form": "rule-following",
                "cause": "instruction-ambiguity",
                "cost": "effort",
                "resolution": "user-correction",
                "leaf": "ignored_constraint",
                "summary": "raised the timeout",
                "evidence": [{"turn": 2, "quote": "I said keep 30s"}],
                "confidence": 1,
            }
        ],
        "abstain": False,
        "notes": "",
    },
    RecoverySpan: {
        "found": True,
        "failure_turn": 1,
        "recovery_start": 2,
        "recovery_end": 3,
        "trigger": "self",
        "rationale": "diagnosed and fixed",
    },
    Attribution: {
        "decisive_turn": 1,
        "cause": "spec",
        "confidence": 1,
        "rationale": "earliest cause",
    },
    EpisodePhases: {
        "phases": [{"kind": "plan", "start_turn": 0, "end_turn": 1, "goal": "explore"}]
    },
    RouterDecision: {"reason": "possible friction", "decision": "needs_analysis"},
    CompiledRules: {
        "rules": [
            {
                "applicability": "when editing parser code",
                "instruction": "run the parser tests",
                "scope": {"task_tags": [], "tools": ["bash"], "paths": []},
                "verifier": {
                    "tier": "semantic",
                    "spec": "a test command was run",
                    "evidence": ["bash"],
                    "failure_message": "run the tests first",
                },
                "rationale": "observed failure",
            }
        ]
    },
}


@pytest.mark.parametrize("schema", list(_SAMPLES))
def test_every_agent_response_validates_strictly(schema: type[BaseModel]) -> None:
    schema.model_validate(_SAMPLES[schema], strict=True)


def test_strict_validation_rejects_a_stringified_number() -> None:
    bad = {**_SAMPLES[JudgeVerdict], "task_adherence": "5"}

    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(bad, strict=True)


def test_strict_validation_rejects_a_stringified_boolean() -> None:
    bad = {**_SAMPLES[ContaminationVerdict], "contaminated": "true"}

    with pytest.raises(ValidationError):
        ContaminationVerdict.model_validate(bad, strict=True)


def test_strict_validation_accepts_a_boolean_where_a_bool_is_expected() -> None:
    verdict = JudgeVerdict.model_validate(_SAMPLES[JudgeVerdict], strict=True)

    assert verdict.accept is True


def test_strict_validation_accepts_an_integral_probability() -> None:
    # JSON has one number type, so `1` is a valid 0-1 probability.
    verdict = DedupVerdict.model_validate(
        {"reason": "r", "duplicate": False, "similarity": 1, "keep": "either"}, strict=True
    )
    assert verdict.similarity == 1.0

    with pytest.raises(ValidationError):
        DedupVerdict.model_validate(
            {"reason": "r", "duplicate": False, "similarity": "1", "keep": "either"},
            strict=True,
        )


def test_strict_validation_accepts_a_str_enum_by_value() -> None:
    rules = CompiledRules.model_validate(_SAMPLES[CompiledRules], strict=True)

    assert rules.rules[0].verifier.tier is VerifierTier.SEMANTIC


_JUDGE_VALUES: dict[str, object] = {
    "candidate_id": "c1",
    "source": "unit",
    "track": "replay",
    "context": "do the thing",
    "completion": "the correct behaviour",
}
_GOOD_JUDGE = (
    '{"reason":"ok","task_adherence":5,"correctness":5,"friction_avoidance":5,'
    '"clarity":5,"evidence":5,"accept":true}'
)
_BAD_JUDGE = (
    '{"reason":"ok","task_adherence":"5","correctness":5,"friction_avoidance":5,'
    '"clarity":5,"evidence":5,"accept":true}'
)


class _ScriptedClient:
    """A stub client that replays a fixed sequence of replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    async def complete(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        force_json: bool = False,
        response_schema: dict[str, object] | None = None,
        schema_name: str = "response",
    ) -> LlmResponse:
        del role, config, messages, tools, force_json, response_schema, schema_name
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return LlmResponse(text=reply, model="stub")


async def test_a_strict_failure_is_repaired_by_the_single_retry() -> None:
    stub = _ScriptedClient([_BAD_JUDGE, _GOOD_JUDGE])
    gateway = LlmGateway(cast("LlmClient", stub), default_registry())

    verdict = await gateway.complete_structured(
        RoleName.JUDGE, LlmRoleConfig(model="stub"), "judge", _JUDGE_VALUES, JudgeVerdict
    )

    assert verdict.accept
    assert stub.calls == 2


async def test_repair_gives_up_after_the_budget() -> None:
    stub = _ScriptedClient([_BAD_JUDGE])
    gateway = LlmGateway(cast("LlmClient", stub), default_registry(), max_repairs=1)

    with pytest.raises(FtValidationError):
        await gateway.complete_structured(
            RoleName.JUDGE, LlmRoleConfig(model="stub"), "judge", _JUDGE_VALUES, JudgeVerdict
        )

    # One initial attempt plus exactly one repair.
    assert stub.calls == 2
