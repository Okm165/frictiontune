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
    assert "[tool result: read]" in chat[1].content
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


class _LastSensitiveTokenizer:
    """A tokenizer whose last assistant turn renders differently, like Qwen3.

    Tokens are characters, so a token span is a character span and the mask can
    be checked exactly. A trailing assistant turn gains a ``THINK`` prefix; a
    non-final one does not.
    """

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in ids)

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False
    ) -> str:
        chunks: list[str] = []
        for index, message in enumerate(messages):
            prefix = ""
            if message.role == "assistant" and index == len(messages) - 1:
                prefix = "THINK"
            chunks.append(f"<{message.role}>{prefix}{message.content}")
        return "".join(chunks)


def test_render_chat_masks_a_middle_assistant_turn_correctly() -> None:
    tokenizer = _LastSensitiveTokenizer()
    messages = [
        ChatMessage.user("q"),
        ChatMessage.assistant("a"),
        ChatMessage(role="tool", content="t"),
    ]

    rendered = render_chat(tokenizer, messages, [False, True, False])

    assert tokenizer.decode(rendered.input_ids) == "<user>q<assistant>a<tool>t"
    # Only the assistant turn is trained; the tool turn after it and the user
    # turn before it are masked.
    assert rendered.assistant_mask == [False] * len("<user>q") + [True] * len("<assistant>a") + [
        False
    ] * len("<tool>t")


def test_render_chat_preserves_the_last_turn_rendering() -> None:
    tokenizer = _LastSensitiveTokenizer()
    messages = [ChatMessage.user("q"), ChatMessage.assistant("a")]
    rendered = render_chat(tokenizer, messages, [False, True])

    assert tokenizer.decode(rendered.input_ids) == "<user>q<assistant>THINKa"


class _ToolMergingTokenizer:
    """A tokenizer that folds consecutive tool turns into one block, like Qwen3."""

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in ids)

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False
    ) -> str:
        chunks: list[str] = []
        for index, message in enumerate(messages):
            if message.role == "tool":
                if index == 0 or messages[index - 1].role != "tool":
                    chunks.append("<tools>")
                chunks.append(message.content)
                if index == len(messages) - 1 or messages[index + 1].role != "tool":
                    chunks.append("</tools>")
            else:
                prefix = (
                    "THINK" if message.role == "assistant" and index == len(messages) - 1 else ""
                )
                chunks.append(f"<{message.role}>{prefix}{message.content}")
        return "".join(chunks)


def test_render_chat_handles_a_run_of_tool_turns() -> None:
    tokenizer = _ToolMergingTokenizer()
    messages = [
        ChatMessage.user("q"),
        ChatMessage.assistant("a"),
        ChatMessage(role="tool", content="t1"),
        ChatMessage(role="tool", content="t2"),
        ChatMessage.assistant("b"),
    ]

    rendered = render_chat(tokenizer, messages, [False, True, False, False, True])

    assert tokenizer.decode(rendered.input_ids) == (tokenizer.apply_chat_template(messages))
    # The two tool turns form one masked block; the assistant turns are trained.
    assert rendered.assistant_mask == (
        [False] * len("<user>q")
        + [True] * len("<assistant>a")
        + [False] * len("<tools>t1t2</tools>")
        + [True] * len("<assistant>THINKb")
    )
