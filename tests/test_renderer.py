"""Tests that SFT rendering preserves the tool-call protocol.

Both tokenizers serialize only a message's role and text content, so a tool call
attached as a structured field would vanish from the training text. These tests
pin that the call and its named result survive into the rendered content.
"""

from __future__ import annotations

from frictiontune.domain.trace import Message, TextPart, ToolPart
from frictiontune.sft.renderer import to_chat_messages
from frictiontune.sft.tokenizer import FallbackTokenizer


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
