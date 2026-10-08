"""Tokenizer abstraction for SFT rendering.

The authoritative tokenizer comes from the base model. Two implementations are
provided: a HuggingFace tokenizer (loaded lazily, only when installed) and a
deterministic fallback so the pipeline is never blocked on a heavyweight
dependency. Both satisfy the same protocol, so ``build_sft`` does not care which
one it has.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from frictiontune.core.errors import BackendUnavailableError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frictiontune.llm.messages import ChatMessage

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_FALLBACK_VOCAB = 1 << 20


@runtime_checkable
class Tokenizer(Protocol):
    """The minimal tokenizer surface the SFT builder needs."""

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token id."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back to text (used for reports)."""
        ...

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False
    ) -> str:
        """Render messages with the model's chat template."""
        ...


def chatml_template(messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False) -> str:
    """A robust ChatML-style fallback template."""
    chunks: list[str] = []
    for message in messages:
        name = f' name="{message.name}"' if message.name else ""
        chunks.append(f"<|im_start|>{message.role}{name}\n{message.content}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


class FallbackTokenizer:
    """A deterministic, dependency-free tokenizer.

    Tokens are word/punctuation units mapped to a large id space by content
    hash. It is lossy and is *not* a substitute for the model's tokenizer; it
    exists so the pipeline and its tests can run without ``transformers``. The
    run manifest records that a fallback was used.
    """

    def __init__(self) -> None:
        """Create a fallback tokenizer with a fixed EOS id."""
        self._eos = 1

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token id."""
        return self._eos

    def _id(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (_FALLBACK_VOCAB - 2) + 2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into deterministic token ids."""
        return [self._id(token) for token in _TOKEN.findall(text)]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode to a placeholder representation (ids are not reversible)."""
        return " ".join(f"<{token_id}>" for token_id in ids)

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False
    ) -> str:
        """Render messages with the ChatML fallback template."""
        return chatml_template(messages, add_generation_prompt=add_generation_prompt)


class HuggingFaceTokenizer:
    """Wraps ``transformers.AutoTokenizer`` for the configured base model."""

    def __init__(self, path: str, *, trust_remote_code: bool = False) -> None:
        """Load the tokenizer for the model at ``path``."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendUnavailableError(
                "tokenizer",
                "install transformers (uv sync --package frictiontune-train --extra train)",
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        if self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token id."""
        eos = self._tokenizer.eos_token_id
        return int(eos) if eos is not None else 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text with the underlying tokenizer."""
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids with the underlying tokenizer."""
        return str(self._tokenizer.decode(list(ids), skip_special_tokens=False))

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = False
    ) -> str:
        """Render messages with the model's real chat template (as text).

        ``name`` and ``tool_call_id`` are forwarded when present: some chat
        templates read them for tool/function messages, and dropping them can
        make a real template fail or mis-render the turn.
        """
        payload: list[dict[str, object]] = []
        for message in messages:
            entry: dict[str, object] = {"role": message.role, "content": message.content}
            if message.name is not None:
                entry["name"] = message.name
            if message.tool_call_id is not None:
                entry["tool_call_id"] = message.tool_call_id
            payload.append(entry)
        rendered = self._tokenizer.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return str(rendered)


def build_tokenizer(
    path: str | None,
    *,
    prefer_hf: bool = True,
    trust_remote_code: bool = False,
) -> tuple[Tokenizer, bool]:
    """Return ``(tokenizer, used_fallback)`` for the configured path.

    Falls back to :class:`FallbackTokenizer` when no path is given or the
    HuggingFace stack is not installed and ``prefer_hf`` is true.
    """
    if path and prefer_hf:
        try:
            return HuggingFaceTokenizer(path, trust_remote_code=trust_remote_code), False
        except BackendUnavailableError:
            return FallbackTokenizer(), True
    return FallbackTokenizer(), True
