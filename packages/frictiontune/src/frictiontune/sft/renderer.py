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
    FileRefPart,
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

# A fixed trailing message appended while measuring one message's token span.
# It makes every real message "not last", which matters for templates such as
# Qwen3 that render the final assistant turn differently (a thinking scaffold).
# A user turn renders identically wherever it appears, so its token block is a
# constant that can be stripped exactly.
_STABILITY_PROBE = ChatMessage.user("frictiontune-stability-probe")

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
    """Serialize a tool result so its name survives templates that drop fields.

    The name is carried as a plain-text marker rather than a ``<tool_response>``
    element: Qwen-style templates already wrap a ``tool`` turn in
    ``<tool_response>``, so emitting the element here would nest it.
    """
    return f"[tool result: {name}]\n{output}"


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


def _text_fragment(part: object) -> str:
    """Render one non-reasoning part to text, or ``""`` for a non-text part."""
    if isinstance(part, FileRefPart):
        suffix = f" {part.description}" if part.description else ""
        return f"[file: {part.path}]{suffix}"
    if isinstance(part, (TextPart, SystemPart, CompactionPart)):
        return part.text
    return ""


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
        elif isinstance(part, ToolPart):
            chunks.append(_render_tool_call(part))
        else:
            fragment = _text_fragment(part)
            if fragment:
                chunks.append(fragment)
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
                fragment
                for fragment in (_text_fragment(part) for part in message.parts)
                if fragment
            )
            # A message that reduces to no content (an idle marker, an empty
            # turn) is dropped rather than emitted as an empty chat turn, which
            # adds no signal and can confuse a template.
            if content:
                chat.append(ChatMessage(role=message.role, content=content))
    return chat


def _encode(tokenizer: Tokenizer, messages: Sequence[ChatMessage]) -> list[int]:
    """Render and tokenize a conversation with the model's chat template."""
    return tokenizer.encode(tokenizer.apply_chat_template(list(messages)), add_special_tokens=False)


def _group_boundaries(messages: Sequence[ChatMessage]) -> list[int]:
    """Indices that split the conversation at append-only template boundaries.

    Qwen-style templates coalesce a run of consecutive ``tool`` turns into a
    single ``<tool_response>`` block, so the boundary between two tool turns is
    not between two template blocks. A run of tool turns shares one trained
    flag, so treating it as one group loses nothing and restores prefix
    stability.
    """
    boundaries = [0]
    index = 0
    while index < len(messages):
        end = index + 1
        if messages[index].role == "tool":
            while end < len(messages) and messages[end].role == "tool":
                end += 1
        boundaries.append(end)
        index = end
    return boundaries


def render_chat(
    tokenizer: Tokenizer,
    messages: Sequence[ChatMessage],
    trained_flags: Sequence[bool],
) -> RenderedChat:
    """Tokenize ``messages`` and mask every token not in a trained message.

    A template may render a message differently depending on what follows it
    (Qwen3 gives the *last* assistant turn a thinking scaffold) and may fold
    several messages into one block (a run of ``tool`` turns). Naive prefix
    tokenization is therefore unsound. Instead a constant probe message is
    appended, which makes every real message "not last"; the probe's fixed
    rendering is stripped to recover the stable token span of each append-only
    group. The final group is taken from the true, un-probed render so its
    last-turn rendering is preserved. A template that rewrites earlier text
    raises rather than mis-masking tokens.
    """
    if len(messages) != len(trained_flags):
        raise ValueError("messages and trained_flags length mismatch")

    ordered = list(messages)
    if not ordered:
        return RenderedChat()

    probe_ids = _encode(tokenizer, [_STABILITY_PROBE])
    probe_length = len(probe_ids)

    full = _encode(tokenizer, ordered)
    boundaries = _group_boundaries(ordered)

    # stable[b] is the rendering of ``ordered[:b]`` with the probe appended and
    # then removed: the append-only token sequence for the first ``b`` messages.
    stable: dict[int, list[int]] = {}
    for boundary in boundaries:
        with_probe = _encode(tokenizer, [*ordered[:boundary], _STABILITY_PROBE])
        if with_probe[-probe_length:] != probe_ids:
            raise ValueError(
                "chat template is not prefix-stable: the probe message did not "
                "render as a fixed trailing block, so the training mask cannot "
                "be trusted"
            )
        stable[boundary] = with_probe[: len(with_probe) - probe_length]

    input_ids: list[int] = []
    assistant_mask: list[bool] = []
    for group in range(len(boundaries) - 1):
        start = boundaries[group]
        end = boundaries[group + 1]
        prefix = stable[start]
        if end < len(ordered):
            extended = stable[end]
            if extended[: len(prefix)] != prefix:
                raise ValueError(
                    "chat template is not prefix-stable: tokenizing a longer "
                    "prefix changed earlier tokens, so the training mask cannot "
                    "be trusted"
                )
            block = extended[len(prefix) :]
        else:
            if full[: len(prefix)] != prefix:
                raise ValueError(
                    "chat template is not prefix-stable: the final message's "
                    "context changed when it became last"
                )
            block = full[len(prefix) :]
        input_ids.extend(block)
        assistant_mask.extend([bool(trained_flags[start])] * len(block))

    if input_ids != full:
        raise ValueError(
            "chat template is not prefix-stable: the per-message spans do not "
            "reconstruct the full conversation"
        )

    return RenderedChat(input_ids=input_ids, assistant_mask=assistant_mask)
