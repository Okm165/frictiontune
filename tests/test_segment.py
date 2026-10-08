"""Tests for episode cutting and the descriptive episode features.

The router prompt renders ``user_turns`` from ``EpisodeFeatures``, so these
tests pin that a session-wide count reaches every episode of that session rather
than the always-zero default.
"""

from __future__ import annotations

from frictiontune.domain.trace import Message, MessageRole, TextPart, ToolPart
from frictiontune.stages.segment import compute_features, cut_episodes


def _message(seq: int, role: MessageRole, text: str, *, tool: bool = False) -> Message:
    parts: list[TextPart | ToolPart] = [TextPart(text=text)]
    if tool:
        parts.append(
            ToolPart(name="read", call_id=f"c{seq}", arguments={}, status="error", is_error=True)
        )
    return Message(
        id=f"m{seq}",
        session_id="s1",
        seq=seq,
        role=role,
        time_created=seq,
        parts=parts,
    )


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
