"""Render canonical messages into a native chat template with a train mask.

Two concerns are kept separate:

* :func:`to_chat_messages` converts domain :class:`Message` objects into
  OpenAI-style :class:`ChatMessage` objects, splitting assistant turns into an
  assistant message plus ``tool`` messages so environment output is never
  trained on.
* :func:`render_chat` tokenizes the conversation and returns, for every token,
  whether it belongs to a trained message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.json import dumps
from frictiontune.domain.trace import (
    CompactionPart,
    Message,
    ReasoningPart,
    SystemPart,
    TextPart,
    ToolPart,
)
from frictiontune.llm.messages import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.sft.tokenizer import Tokenizer

_REASONING_OPEN = "<|reasoning|>"
_REASONING_CLOSE = "<|/reasoning|>"

# Both tokenizers serialize only a message's role and text content, so a tool
# call attached as a structured field would be silently dropped from the
# training text (and a tool result's name and id with it). These markers carry
# the call and its result inside the content itself, which every chat template
# renders, so the trained sequence contains the tool interaction.
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _render_tool_call(part: ToolPart) -> str:
    """Serialize one tool call into the assistant content as tagged JSON."""
    payload = dumps({"name": part.name, "arguments": part.arguments}, sort_keys=False)
    return f"{_TOOL_CALL_OPEN}{payload}{_TOOL_CALL_CLOSE}"


def _tool_response_text(name: str, output: str) -> str:
    """Serialize a tool result so its name survives templates that drop fields."""
    return f'<tool_response name="{name}">\n{output}\n</tool_response>'


class RenderedChat(BaseModel):
    """A tokenized conversation plus the per-token training mask."""

    model_config = ConfigDict(extra="forbid")

    input_ids: list[int] = Field(default_factory=list)
    assistant_mask: list[bool] = Field(default_factory=list)

    def trained_tokens(self) -> int:
        """Number of tokens marked for training."""
        return sum(1 for flag in self.assistant_mask if flag)

    def check(self) -> None:
        """Assert the token sequence and mask have equal length."""
        if len(self.input_ids) != len(self.assistant_mask):
            raise ValueError("input_ids and assistant_mask length mismatch")


def _assistant_content(message: Message, *, include_reasoning: bool) -> str:
    """Render an assistant message, preserving text, reasoning, and tool calls.

    Parts are rendered in their original order; tool calls appear as tagged JSON
    at the point they were made, so the trained sequence keeps the agent's tool
    protocol rather than dropping it.
    """
    chunks: list[str] = []
    for part in message.parts:
        if isinstance(part, ReasoningPart):
            if include_reasoning and part.text:
                chunks.append(f"{_REASONING_OPEN}\n{part.text}\n{_REASONING_CLOSE}")
        elif isinstance(part, (TextPart, SystemPart, CompactionPart)) and part.text:
            chunks.append(part.text)
        elif isinstance(part, ToolPart):
            chunks.append(_render_tool_call(part))
    return "\n".join(chunks).strip()


def to_chat_messages(
    messages: Sequence[Message],
    *,
    include_reasoning: bool,
) -> list[ChatMessage]:
    """Convert domain messages to chat messages, expanding tool calls/results.

    Tool calls are embedded in the assistant content (see :func:`_assistant_content`);
    tool results become ``role="tool"`` messages whose content carries the tool
    name, so a template that ignores structured fields still preserves the link.
    Callers always mask the tool-result messages.
    """
    chat: list[ChatMessage] = []
    for message in messages:
        if message.role == "assistant":
            content = _assistant_content(message, include_reasoning=include_reasoning)
            results: list[ChatMessage] = []
            for index, part in enumerate(message.parts):
                if not isinstance(part, ToolPart):
                    continue
                call_id = part.call_id or f"{message.id}:tool{index}"
                if part.output is not None:
                    results.append(
                        ChatMessage(
                            role="tool",
                            content=_tool_response_text(part.name, part.output),
                            name=part.name,
                            tool_call_id=call_id,
                        )
                    )
            if content:
                chat.append(ChatMessage(role="assistant", content=content))
            chat.extend(results)
        else:
            content = "\n".join(
                part.text
                for part in message.parts
                if isinstance(part, (TextPart, SystemPart, CompactionPart)) and part.text
            )
            chat.append(ChatMessage(role=message.role, content=content))
    return chat


def render_chat(
    tokenizer: Tokenizer,
    messages: Sequence[ChatMessage],
    trained_flags: Sequence[bool],
) -> RenderedChat:
    """Tokenize ``messages`` and mask every token not in a trained message.

    The template is applied incrementally: each step re-renders the prefix and
    keeps only the newly appended tokens. This is correct for the standard
    monotonic chat templates (ChatML, Llama, Qwen) and is O(n²) in message count,
    which is negligible next to model calls.
    """
    if len(messages) != len(trained_flags):
        raise ValueError("messages and trained_flags length mismatch")

    input_ids: list[int] = []
    assistant_mask: list[bool] = []
    previous_length = 0
    for index, _message in enumerate(messages):
        rendered = tokenizer.apply_chat_template(list(messages[: index + 1]))
        current = tokenizer.encode(rendered, add_special_tokens=False)
        appended = current[previous_length:]
        input_ids.extend(appended)
        assistant_mask.extend([bool(trained_flags[index])] * len(appended))
        previous_length = len(current)

    return RenderedChat(input_ids=input_ids, assistant_mask=assistant_mask)
