"""Chat message models for the OpenAI-compatible interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from frictiontune.core.json import JsonObject

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A model-emitted tool invocation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: JsonObject = Field(default_factory=dict)

    def to_openai(self) -> dict[str, object]:
        """Render as an OpenAI ``tool_calls`` entry."""
        from frictiontune.core.json import dumps

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": dumps(self.arguments)},
        }


class ToolSpec(BaseModel):
    """A tool exposed to a model (JSON-Schema parameters)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: JsonObject = Field(default_factory=dict)

    def to_openai(self) -> dict[str, object]:
        """Render as an OpenAI function-tool entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ChatMessage(BaseModel):
    """One message exchanged with a model."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def to_openai(self) -> dict[str, object]:
        """Render as an OpenAI chat-completions message."""
        payload: dict[str, object] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return payload

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        """Construct a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        """Construct a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        """Construct an assistant message."""
        return cls(role="assistant", content=content)
