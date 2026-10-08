"""Tests for episode cutting and the descriptive episode features.

The router prompt renders ``user_turns`` from ``EpisodeFeatures``, so these
tests pin that a session-wide count reaches every episode of that session rather
than the always-zero default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from frictiontune.core.artifacts import DatasetRef
from frictiontune.core.errors import StageError
from frictiontune.domain.trace import Message, MessageRole, TextPart, ToolPart
from frictiontune.pipeline.context import ArtifactCatalog, RunContext, StageContext
from frictiontune.stages.segment import Segment, compute_features, cut_episodes

if TYPE_CHECKING:
    from pathlib import Path


def _message(
    seq: int, role: MessageRole, text: str, *, tool: bool = False, session_id: str = "s1"
) -> Message:
    parts: list[TextPart | ToolPart] = [TextPart(text=text)]
    if tool:
        parts.append(
            ToolPart(name="read", call_id=f"c{seq}", arguments={}, status="error", is_error=True)
        )
    return Message(
        id=f"m{seq}",
        session_id=session_id,
        seq=seq,
        role=role,
        time_created=seq,
        parts=parts,
    )


def _segment_context(tmp_path: Path, messages: list[Message]) -> StageContext:
    path = tmp_path / "run" / "data" / "messages.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(message.model_dump_json() for message in messages) + "\n")
    catalog = ArtifactCatalog()
    catalog.publish(
        DatasetRef(name="messages", path=str(path), sha256="x", rows=len(messages), kind="jsonl"),
        kind="jsonl",
    )
    run = RunContext(
        config=cast("Any", object()),
        pipeline="core",
        cache=cast("Any", object()),
        gateway=cast("Any", object()),
        backends=cast("Any", object()),
        logger=cast("Any", object()),
        artifacts_root=tmp_path,
        catalog=catalog,
    )
    return StageContext(run=run, stage="segment", run_id="r", store=cast("Any", object()))


def test_user_turns_in_session_reaches_every_episode() -> None:
    messages = [
        _message(0, "user", "first"),
        _message(1, "assistant", "a", tool=True),
        _message(2, "user", "second"),
        _message(3, "assistant", "b"),
        _message(4, "user", "third"),
        _message(5, "assistant", "c"),
    ]

    episodes = cut_episodes("s1", messages)

    assert len(episodes) == 3
    assert [episode.features.user_turns_in_session for episode in episodes] == [3, 3, 3]


def test_cut_episodes_starts_at_each_user_message() -> None:
    messages = [
        _message(0, "user", "first"),
        _message(1, "assistant", "a"),
        _message(2, "user", "second"),
        _message(3, "assistant", "b"),
    ]

    episodes = cut_episodes("s1", messages)

    assert [episode.start_user_message_id for episode in episodes] == ["m0", "m2"]
    assert [len(episode.messages) for episode in episodes] == [2, 2]


def test_compute_features_counts_tools_and_errors() -> None:
    features = compute_features([_message(0, "assistant", "a", tool=True)], user_turns_in_session=7)

    assert features.turns == 1
    assert features.tool_calls == 1
    assert features.tool_errors == 1
    assert features.user_turns_in_session == 7


def test_a_leading_assistant_turn_without_a_user_is_kept() -> None:
    messages = [_message(0, "assistant", "preamble"), _message(1, "user", "hi")]

    episodes = cut_episodes("s1", messages)

    # The assistant preamble has no user boundary, so it is dropped; the episode
    # starts at the real user turn.
    assert [episode.start_user_message_id for episode in episodes] == ["m1"]


def test_group_messages_yields_each_contiguous_session(tmp_path: Path) -> None:
    messages = [
        _message(0, "user", "a", session_id="s1"),
        _message(1, "assistant", "b", session_id="s1"),
        _message(2, "user", "c", session_id="s2"),
        _message(3, "assistant", "d", session_id="s2"),
    ]

    groups = list(Segment()._group_messages(_segment_context(tmp_path, messages)))

    assert [session_id for session_id, _ in groups] == ["s1", "s2"]
    assert [len(items) for _, items in groups] == [2, 2]


def test_group_messages_rejects_a_non_contiguous_session(tmp_path: Path) -> None:
    messages = [
        _message(0, "user", "a", session_id="s1"),
        _message(1, "user", "b", session_id="s2"),
        _message(2, "assistant", "c", session_id="s1"),
    ]

    with pytest.raises(StageError, match="not contiguous"):
        list(Segment()._group_messages(_segment_context(tmp_path, messages)))
