"""LLM access: one transport client, one gateway, one prompt registry.

Every LLM call in the codebase goes through :class:`LlmGateway`.
"""

from frictiontune.llm.client import LlmClient, LlmResponse, TokenUsage
from frictiontune.llm.gateway import LlmGateway
from frictiontune.llm.messages import ChatMessage, ToolCall, ToolSpec
from frictiontune.llm.parsing import extract_json_object
from frictiontune.llm.prompts.registry import PromptRegistry, PromptTemplate, default_registry

__all__ = [
    "ChatMessage",
    "LlmClient",
    "LlmGateway",
    "LlmResponse",
    "PromptRegistry",
    "PromptTemplate",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    "default_registry",
    "extract_json_object",
]
