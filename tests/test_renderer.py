"""Tests that SFT rendering preserves the tool-call protocol.

Both tokenizers serialize only a message's role and text content, so a tool call
attached as a structured field would vanish from the training text. These tests
pin that the call and its named result survive into the rendered content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from frictiontune.domain.trace import FileRefPart, IdlePart, Message, TextPart, ToolPart
from frictiontune.llm.messages import ChatMessage
from frictiontune.sft.renderer import render_chat, to_chat_messages
from frictiontune.sft.tokenizer import FallbackTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence


def _assistant_with_tool(*, with_output: bool = True) -> Message:
    return Message(
        id="m1",
        session_id="s",
        seq=1,
        role="assistant",
        time_created=0,
        parts=[
            TextPart(text="I will read the file."),
            ToolPart(
                name="read",
                call_id="c1",
                arguments={"path": "a.py"},
                status="ok",
                output="print(1)" if with_output else None,
            ),
        ],
    )


def test_tool_call_is_rendered_into_the_assistant_content() -> None:
    chat = to_chat_messages([_assistant_with_tool()], include_reasoning=False)

    assert chat[0].role == "assistant"
    assert "<tool_call>" in chat[0].content
    assert '"name":"read"' in chat[0].content
    assert '"path":"a.py"' in chat[0].content
    assert "I will read the file." in chat[0].content


def test_tool_result_carries_its_name_and_output() -> None:
    chat = to_chat_messages([_assistant_with_tool()], include_reasoning=False)

    assert chat[1].role == "tool"
    assert 'name="read"' in chat[1].content
    assert "print(1)" in chat[1].content


def test_tool_call_without_output_still_renders_the_call() -> None:
    chat = to_chat_messages([_assistant_with_tool(with_output=False)], include_reasoning=False)

    assert len(chat) == 1
    assert "<tool_call>" in chat[0].content


def test_fallback_template_renders_the_whole_tool_protocol() -> None:
    chat = to_chat_messages([_assistant_with_tool()], include_reasoning=False)

    rendered = FallbackTokenizer().apply_chat_template(chat)

    assert "<tool_call>" in rendered
    assert "</tool_call>" in rendered
    assert "print(1)" in rendered


def test_file_reference_is_rendered_into_content() -> None:
    message = Message(
        id="m0",
        session_id="s",
        seq=0,
        role="user",
        time_created=0,
        parts=[TextPart(text="see"), FileRefPart(path="a.py", description="the file")],
    )

    chat = to_chat_messages([message], include_reasoning=False)

    assert chat[0].content == "see\n[file: a.py] the file"


def test_a_message_with_no_content_is_dropped() -> None:
    message = Message(
        id="m0", session_id="s", seq=0, role="user", time_created=0, parts=[IdlePart()]
    )

    assert to_chat_messages([message], include_reasoning=False) == []


class _NonMonotonicTokenizer:
    """A deliberately prefix-unstable tokenizer, to prove the guard fires."""

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text)]

    def decode(self, ids: Sequence[int]) -> str:
        del ids
        return ""

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False
    ) -> str:
        del add_generation_prompt
        return "x" * len(messages)


def test_render_chat_rejects_a_non_prefix_stable_template() -> None:
    messages = [ChatMessage.user("a"), ChatMessage.assistant("b")]

    with pytest.raises(ValueError, match="prefix-stable"):
        render_chat(_NonMonotonicTokenizer(), messages, [False, True])
