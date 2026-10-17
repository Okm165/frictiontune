"""Tests for recovery extraction: one example per recovery span.

``mine`` asks the recovery finder once per friction event, but an episode with
several events often has one recovery that resolves them all. Writing that
example once per event filled 42% of the first real SFT pool with literal
copies — byte-identical ``(context, completion)`` — and bought a judge call for
each, so the recovery span is the identity of an example and a repeat is
dropped at the source.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from frictiontune.config.models import LlmRoleConfig
from frictiontune.domain.episodes import Episode
from frictiontune.domain.friction import EvidenceSpan, FrictionEvent, RecoverySpan
from frictiontune.domain.responses import HarvestedSpan, HarvestedWork
from frictiontune.domain.trace import Message, MessageRole, TextPart
from frictiontune.stages.mine import Mine, MineSettings

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Gateway:
    """A stub miner: canned recovery spans, and one canned harvest reply."""

    def __init__(self, spans: dict[str, RecoverySpan], work: HarvestedWork | None = None) -> None:
        self._spans = spans
        self._work = work if work is not None else HarvestedWork()

    async def complete_structured(
        self,
        role: object,
        config: object,
        prompt: object,
        values: dict[str, Any],
        schema: object,
    ) -> Any:
        if prompt == "harvest":
            return self._work
        return self._spans[values["friction_summary"]]


def _message(seq: int, role: MessageRole, text: str) -> Message:
    """Build one canonical message for the episode transcript."""
    return Message(
        id=f"m{seq}",
        session_id="s1",
        seq=seq,
        role=role,
        time_created=seq,
        parts=[TextPart(text=text)],
    )


def _event(event_id: str) -> FrictionEvent:
    """Build one persisted friction event on the episode."""
    return FrictionEvent(
        id=event_id,
        episode_id="e1",
        leaf="ignored_constraint",
        cost="effort",
        resolution="self-recovery",
        summary=event_id,
        evidence=[EvidenceSpan(turn=0, quote="q")],
        confidence=0.9,
        form="rule-following",
        cause="instruction-ambiguity",
    )


def _run(
    spans: dict[str, RecoverySpan],
    work: HarvestedWork | None = None,
    *,
    events: list[FrictionEvent] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Run ``mine`` over one episode; return the writes and the reports."""
    episode = Episode(
        id="e1",
        session_id="s1",
        start_user_message_id="m0",
        start_seq=0,
        end_seq=3,
        messages=[
            _message(0, "user", "do it"),
            _message(1, "assistant", "done"),
            _message(2, "user", "not like that"),
            _message(3, "assistant", "fixed"),
        ],
    )
    events = events if events is not None else [_event("f1"), _event("f2"), _event("f3")]
    written: list[Any] = []
    sink = SimpleNamespace(write=written.append, close=lambda: None)
    config = LlmRoleConfig(model="stub")
    taxonomy = SimpleNamespace(
        leaves={
            "ignored_constraint": SimpleNamespace(
                form="rule-following", cause="instruction-ambiguity", description="d"
            )
        }
    )

    def stream(name: str, model: object) -> Iterator[Any]:
        if name == "episodes":
            return iter([episode])
        if name == "friction_events":
            return iter(events)
        return iter([])

    reports: dict[str, Any] = {}
    ctx = cast(
        "Any",
        SimpleNamespace(
            stage="mine",
            run_id="r",
            outputs={},
            settings=lambda model: MineSettings(),
            config=SimpleNamespace(role=lambda name: config, taxonomy=taxonomy),
            run=SimpleNamespace(gateway=_Gateway(spans, work)),
            catalog=SimpleNamespace(stream=stream, has=lambda name: False),
            jsonl_sink=lambda name: sink,
            emit_json=lambda name, value: reports.__setitem__(name, value),
            emit_text=lambda name, text: None,
        ),
    )
    asyncio.run(Mine().run(ctx))
    return written, reports


def test_events_sharing_a_recovery_span_yield_one_candidate() -> None:
    spans = {
        "f1": RecoverySpan(found=True, recovery_start=1, recovery_end=2, trigger="self"),
        "f2": RecoverySpan(found=True, recovery_start=1, recovery_end=2, trigger="self"),
        "f3": RecoverySpan(found=True, recovery_start=2, recovery_end=3, trigger="self"),
    }

    written, reports = _run(spans)

    assert [example.meta["event_id"] for example in written] == ["f1", "f3"]
    assert reports["mine_report"]["candidates"] == 2
    assert reports["mine_report"]["skip_reasons"]["duplicate_span"] == 1


def test_events_with_distinct_spans_all_survive() -> None:
    spans = {
        "f1": RecoverySpan(found=True, recovery_start=1, recovery_end=1, trigger="self"),
        "f2": RecoverySpan(found=True, recovery_start=2, recovery_end=3, trigger="self"),
        "f3": RecoverySpan(found=True, recovery_start=3, recovery_end=3, trigger="self"),
    }

    written, reports = _run(spans)

    assert [example.meta["event_id"] for example in written] == ["f1", "f2", "f3"]
    assert reports["mine_report"]["skip_reasons"].get("duplicate_span") is None


def test_harvest_emits_focused_good_work_as_a_success_example() -> None:
    # Recovery spans cover turns 1 and 2, leaving turn 3 free to harvest.
    spans = {
        "f1": RecoverySpan(found=True, recovery_start=1, recovery_end=1, trigger="self"),
        "f2": RecoverySpan(found=True, recovery_start=1, recovery_end=1, trigger="self"),
        "f3": RecoverySpan(found=True, recovery_start=2, recovery_end=2, trigger="self"),
    }
    work = HarvestedWork(
        spans=[HarvestedSpan(start=3, end=3, behaviour="stated the fix plainly")],
        notes="",
    )

    written, reports = _run(spans, work)

    harvest = [e for e in written if e.track == "success"]
    assert len(harvest) == 1
    example = harvest[0]
    assert example.meta["behaviour"] == "stated the fix plainly"
    assert example.meta["harvest_start"] == 3
    assert [m.role for m in example.completion] == ["assistant"]
    assert [m.role for m in example.context] == ["user", "assistant", "user"]
    assert reports["mine_report"]["harvest"]["examples"] == 1
    assert reports["mine_report"]["harvest"]["episodes_with_spans"] == 1


def test_harvest_skips_a_span_the_recovery_already_covers() -> None:
    spans = {
        "f1": RecoverySpan(found=True, recovery_start=1, recovery_end=1, trigger="self"),
        "f2": RecoverySpan(found=True, recovery_start=1, recovery_end=1, trigger="self"),
        "f3": RecoverySpan(found=True, recovery_start=2, recovery_end=3, trigger="self"),
    }
    # Turn 3 is inside the recovery span mined for f3, so the harvest pass must
    # not emit it again: the behaviour is already in the pool.
    work = HarvestedWork(spans=[HarvestedSpan(start=3, end=3, behaviour="duplicate")])

    written, reports = _run(spans, work)

    assert [e for e in written if e.track == "success"] == []
    assert reports["mine_report"]["harvest"]["examples"] == 0


def test_harvest_drops_a_span_longer_than_the_turn_budget() -> None:
    # Four turns exceeds the budget of three; three turns is the documented
    # maximum and must survive.
    work = HarvestedWork(
        spans=[
            HarvestedSpan(start=0, end=3, behaviour="a whole stretch of work"),
            HarvestedSpan(start=1, end=3, behaviour="three turns, at the limit"),
        ]
    )

    written, _reports = _run({}, work, events=[])

    harvest = [e for e in written if e.track == "success"]
    assert [e.meta["behaviour"] for e in harvest] == ["three turns, at the limit"]


def test_harvest_runs_on_episodes_with_no_friction() -> None:
    # A clean episode is exactly where good work is least interrupted, so the
    # pass must not be gated on friction events.
    work = HarvestedWork(spans=[HarvestedSpan(start=1, end=1, behaviour="checked first")])

    written, reports = _run({}, work, events=[])

    assert [e.track for e in written] == ["success"]
    assert reports["mine_report"]["candidates"] == 0
    assert reports["mine_report"]["harvest"]["examples"] == 1
