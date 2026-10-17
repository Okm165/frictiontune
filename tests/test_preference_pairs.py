"""Tests for the preference-pair stage's deterministic plumbing (GAP-104).

The pairs are plumbing over judgments the pipeline already made: the friction
event's evidence points at the failing turns, the recovery example carries the
chosen continuation, and the construction is mechanical. No LLM, no GPU.
"""

from __future__ import annotations

from frictiontune.domain.dataset import PreferencePair, TrainingExample
from frictiontune.domain.friction import EvidenceSpan, FrictionEvent
from frictiontune.domain.trace import Message, TextPart
from frictiontune.stages._extensions import (
    PreferencePairsSettings,
    _pair_from_example,
)


def _message(identifier: str, role: str, text: str) -> Message:
    return Message(
        id=identifier,
        session_id="ses_x",
        seq=0,
        role=role,
        time_created=0,
        parts=[TextPart(text=text)],
    )


def _example(context: list[Message], completion: list[Message]) -> TrainingExample:
    return TrainingExample(
        id="ex1",
        source="ses_x:ep1",
        track="recovery",
        context=context,
        completion=completion,
        meta={"event_id": "ev1", "session_id": "ses_x"},
    )


def _event(turn: int) -> FrictionEvent:
    return FrictionEvent(
        id="ev1",
        episode_id="ses_x:ep1",
        leaf="unverified_claim",
        cost="medium",
        resolution="self-recovery",
        evidence=[EvidenceSpan(turn=turn, quote="the failing turn")],
        confidence=0.9,
        form="act",
        cause="unverified",
    )


def _assistant(identifier: str, text: str) -> Message:
    return _message(identifier, "assistant", text)


def _user(identifier: str, text: str) -> Message:
    return _message(identifier, "user", text)


def _pair(
    context: list[Message], completion: list[Message], turn: int = 2
) -> tuple[PreferencePair | None, str]:
    return _pair_from_example(
        _example(context, completion), _event(turn), PreferencePairsSettings()
    )


def test_pair_isolates_the_failing_turns_and_keeps_state_in_the_prompt() -> None:
    context = [
        _user("m1", "run the tests"),
        _assistant("m2", "running"),  # pre-friction ack (before evidence turn)
        _assistant("m3", "all tests pass " + "and verified " * 20),  # friction (turn 2)
        _user("m4", "they clearly fail"),  # the user's correction stays in the prompt
    ]
    completion = [_assistant("m5", "you are right; here is the failing output")]

    pair, reason = _pair(context, completion, turn=2)

    assert reason == "paired" and pair is not None
    # prompt = pre-friction context + the non-assistant correction
    assert [m.id for m in pair.prompt] == ["m1", "m2", "m4"]
    # rejected = the assistant's failing behaviour, verbatim
    assert [m.id for m in pair.rejected] == ["m3"]
    # chosen = the recovery the session actually produced
    assert [m.id for m in pair.chosen] == ["m5"]


def test_a_thin_rejected_side_is_skipped() -> None:
    context = [
        _user("m1", "go"),
        _assistant("m2", "ok"),
        _assistant("m3", "done"),
    ]
    pair, reason = _pair(context, [_assistant("m4", "redo")], turn=2)

    assert pair is None
    assert reason == "rejected_too_thin"


def test_friction_beyond_the_context_is_skipped_not_crashed() -> None:
    context = [_user("m1", "go"), _assistant("m2", "doing")]
    pair, reason = _pair(context, [_assistant("m3", "fix")], turn=9)

    assert pair is None
    assert reason == "friction_beyond_context"


def test_a_missing_event_is_reported_not_fatal() -> None:
    pair, reason = _pair_from_example(
        _example([_user("m1", "go")], [_assistant("m2", "fix")]),
        None,
        PreferencePairsSettings(),
    )
    assert pair is None
    assert reason == "event_not_found"


def test_prompt_trims_from_the_oldest_end_to_the_budget() -> None:
    context = [_user(f"m{i}", "z" * 2_000) for i in range(20)] + [
        _assistant("friction", "f" * 1_000)
    ]
    completion = [_assistant("fix", "r" * 1_000)]

    pair, reason = _pair(context, completion, turn=20)

    assert reason == "paired" and pair is not None
    total = sum(len(m.content_text()) for m in pair.prompt)
    assert total <= PreferencePairsSettings().max_pair_context_chars
    # the newest PRE-FRICTION state survives the trim; the failing turn
    # itself is the rejected side, not the prompt
    assert pair.prompt[-1].id == "m19"
    assert pair.rejected[0].id == "friction"


def test_examples_outside_the_training_split_never_become_pairs() -> None:
    # `judged` contains the held-out episodes too; a pair built from one would
    # train the DPO adapter on the test set's failures. The guard is the split,
    # and there is no bypass.
    context = [
        _user("m1", "run the tests"),
        _assistant("m2", "all tests pass " + "and verified " * 20),
    ]
    completion = [_assistant("m3", "here is the failing output")]

    guarded, reason = _pair_from_example(
        _example(context, completion),
        _event(turn=1),
        PreferencePairsSettings(),
        train_ids={"some_other_example"},
    )
    assert guarded is None
    assert reason == "not_in_train_split"

    allowed, reason = _pair_from_example(
        _example(context, completion),
        _event(turn=1),
        PreferencePairsSettings(),
        train_ids={"ex1"},
    )
    assert reason == "paired" and allowed is not None


def test_low_confidence_events_are_droppable() -> None:
    context = [
        _user("m1", "run the tests"),
        _assistant("m2", "all tests pass " + "and verified " * 20),
    ]
    completion = [_assistant("m3", "here is the failing output")]
    event = _event(turn=1)

    kept, _ = _pair_from_example(
        _example(context, completion), event, PreferencePairsSettings(), train_ids={"ex1"}
    )
    assert kept is not None

    strict, reason = _pair_from_example(
        _example(context, completion),
        event,
        PreferencePairsSettings(min_event_confidence=0.95),
        train_ids={"ex1"},
    )
    assert strict is None
    assert reason == "low_confidence"


def test_preference_pairs_stage_consumes_the_split() -> None:
    from frictiontune.pipeline.dag import full_graph
    from frictiontune.stages._extensions import BuildPreferencePairs

    graph = full_graph()
    # The stage must run AFTER the split exists: before it, the leakage guard
    # has nothing to guard with.
    assert graph.predecessors("build_preference_pairs") == ("decontaminate",)
    assert "train" in BuildPreferencePairs.spec.inputs
