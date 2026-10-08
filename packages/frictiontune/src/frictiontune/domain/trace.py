"""Canonical conversation trace: sessions, messages, and typed parts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.json import JsonObject, dumps

_TOOL_ARGS_LIMIT = 400


def _inline_arguments(arguments: JsonObject) -> str:
    """Render tool arguments compactly, bounded so a large payload cannot crowd out output."""
    rendered = dumps(arguments)
    if len(rendered) <= _TOOL_ARGS_LIMIT:
        return rendered
    return rendered[:_TOOL_ARGS_LIMIT] + f"… [+{len(rendered) - _TOOL_ARGS_LIMIT} chars]"


class DomainModel(BaseModel):
    """Strict base for every domain object."""

    model_config = ConfigDict(extra="forbid")


class ReasoningPart(DomainModel):
    """The model's private reasoning for a turn."""

    kind: Literal["reasoning"] = "reasoning"
    text: str


class TextPart(DomainModel):
    """Visible assistant or user prose."""

    kind: Literal["text"] = "text"
    text: str


class ToolPart(DomainModel):
    """A tool call and, when present, its result."""

    kind: Literal["tool"] = "tool"
    name: str
    call_id: str | None = None
    arguments: JsonObject = Field(default_factory=dict)
    status: str | None = None
    output: str | None = None
    is_error: bool = False


class CompactionPart(DomainModel):
    """A context-compaction marker emitted by the runtime (context-rot proxy)."""

    kind: Literal["compaction"] = "compaction"
    text: str = ""


class SystemPart(DomainModel):
    """A system/meta turn."""

    kind: Literal["system"] = "system"
    text: str = ""


class FileRefPart(DomainModel):
    """A file the user attached or referenced."""

    kind: Literal["file"] = "file"
    path: str
    description: str = ""


class IdlePart(DomainModel):
    """An idle/no-content marker."""

    kind: Literal["idle"] = "idle"


Part = Annotated[
    ReasoningPart | TextPart | ToolPart | CompactionPart | SystemPart | FileRefPart | IdlePart,
    Field(discriminator="kind"),
]

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(DomainModel):
    """One canonical message in a session."""

    id: str
    session_id: str
    seq: int
    role: MessageRole
    time_created: int
    parts: list[Part] = Field(default_factory=list)
    agent: str | None = None
    model: str | None = None
    raw: JsonObject | None = None

    def content_text(self) -> str:
        """Concatenate the text-bearing parts (reasoning + text + tool output)."""
        chunks: list[str] = []
        for part in self.parts:
            if isinstance(part, (ReasoningPart, TextPart, SystemPart, CompactionPart)):
                if part.text:
                    chunks.append(part.text)
            elif isinstance(part, ToolPart):
                call = f"[tool:{part.name}"
                if part.arguments:
                    call += f" {_inline_arguments(part.arguments)}"
                call += "]"
                chunks.append(f"{call} {part.output or ''}".rstrip())
        return "\n".join(chunks)

    def has_tool_error(self) -> bool:
        """Whether any tool part failed."""
        return any(isinstance(part, ToolPart) and part.is_error for part in self.parts)


class Session(DomainModel):
    """A whole agent session."""

    id: str
    project: str | None = None
    model: str | None = None
    agent: str | None = None
    title: str | None = None
    time_created: int = 0
