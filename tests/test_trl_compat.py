"""Tests for the TRL version adapter.

The trainer call sites must resolve argument names against the installed TRL,
not assume one release's spelling. These tests pin that behaviour with fake
classes, so a rename is caught without importing torch or trl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from frictiontune.core.errors import StageError
from frictiontune_train.trl_compat import resolve_config_kwargs, trainer_token_kwargs


@dataclass
class _NewConfig:
    """A config with current TRL field names plus a non-init field."""

    output_dir: str
    max_length: int = 1024
    dataset_kwargs: dict[str, Any] | None = None
    _internal: int = field(default=0, init=False)


@dataclass
class _OldConfig:
    """A config with the pre-rename sequence-length field."""

    max_seq_length: int = 512


@dataclass
class _OpenAiUnsupportedConfig:
    """A config lacking any generation-length field."""

    output_dir: str = ""


class _ModernTrainer:
    """A trainer exposing ``processing_class`` (TRL >= 0.12)."""

    def __init__(
        self, model: object, *, processing_class: object = None, args: object = None
    ) -> None:
        """Accept the current tokenizer kwarg."""


class _LegacyTrainer:
    """A trainer exposing the old ``tokenizer`` kwarg."""

    def __init__(self, model: object, *, tokenizer: object = None, args: object = None) -> None:
        """Accept the legacy tokenizer kwarg."""


class _NoTokenTrainer:
    """A trainer that declares no tokenizer argument at all."""

    def __init__(self, model: object) -> None:
        """Declare no tokenizer kwarg."""


def test_resolve_prefers_the_renamed_field() -> None:
    resolved = resolve_config_kwargs(
        _NewConfig,
        {"max_seq_length": 2048},
        aliases={"max_seq_length": ("max_length", "max_seq_length")},
        stage="train_sft",
    )

    assert resolved == {"max_length": 2048}


def test_resolve_falls_back_to_the_legacy_field() -> None:
    resolved = resolve_config_kwargs(
        _OldConfig,
        {"max_seq_length": 2048},
        aliases={"max_seq_length": ("max_length", "max_seq_length")},
        stage="train_sft",
    )

    assert resolved == {"max_seq_length": 2048}


def test_resolve_raises_when_no_field_name_matches() -> None:
    with pytest.raises(StageError, match="max_completion_length"):
        resolve_config_kwargs(
            _OpenAiUnsupportedConfig,
            {"max_new_tokens": 64},
            aliases={"max_new_tokens": ("max_completion_length",)},
            stage="train_grpo",
        )


def test_resolve_ignores_non_initialised_fields() -> None:
    with pytest.raises(StageError, match="_internal"):
        resolve_config_kwargs(_NewConfig, {"_internal": 1}, stage="train_sft")


def test_trainer_token_kwarg_prefers_processing_class() -> None:
    tokenizer = object()

    assert trainer_token_kwargs(_ModernTrainer, tokenizer, stage="train_sft") == {
        "processing_class": tokenizer
    }


def test_trainer_token_kwarg_falls_back_to_tokenizer() -> None:
    tokenizer = object()

    assert trainer_token_kwargs(_LegacyTrainer, tokenizer, stage="train_sft") == {
        "tokenizer": tokenizer
    }


def test_trainer_token_kwarg_raises_without_a_token_argument() -> None:
    with pytest.raises(StageError, match="processing_class"):
        trainer_token_kwargs(_NoTokenTrainer, object(), stage="train_sft")
