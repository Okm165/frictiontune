"""Tests for the IO on both sides of the model boundary.

A corrupt base-model config or a half-written SFT artifact must fail with a
named stage error, not a bare ``json.JSONDecodeError`` traceback, so a real run
diagnoses the cause instead of the symptom.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from frictiontune.core.errors import StageError
from frictiontune.stages.fetch_base import probe_base
from frictiontune_train.sft import _read_records, _select_losses

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(repo_dir: Path, payload: object) -> None:
    (repo_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_probe_base_reads_a_standard_config(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
        },
    )

    probe = probe_base(tmp_path, "meta-llama/Llama-3")

    assert probe.architecture == "LlamaForCausalLM"
    assert probe.hidden_size == 4096
    assert probe.trainable is True


def test_probe_base_rejects_malformed_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(StageError, match=r"unreadable config\.json"):
        probe_base(tmp_path, "x")


def test_probe_base_rejects_non_object_config(tmp_path: Path) -> None:
    _write_config(tmp_path, [1, 2, 3])

    with pytest.raises(StageError, match=r"invalid config\.json"):
        probe_base(tmp_path, "x")


def test_probe_base_flags_gguf_as_untrainable(tmp_path: Path) -> None:
    _write_config(tmp_path, {"model_type": "llama"})
    (tmp_path / "model.gguf").write_bytes(b"\x00")

    probe = probe_base(tmp_path, "x")

    assert probe.trainable is False
    assert any("GGUF" in note for note in probe.notes)


def test_read_records_raises_a_named_error_on_a_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text('{"input_ids": [1], "labels": [-100]}\n{bad json\n', encoding="utf-8")

    with pytest.raises(StageError, match="malformed JSON line"):
        _read_records(path)


def test_read_records_raises_a_named_error_on_a_non_object_record(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(StageError, match="has no input_ids/labels"):
        _read_records(path)


def test_read_records_keeps_only_tokens_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"id": "a", "input_ids": [1, 2], "labels": [-100, 2], "meta": {"x": 1}}\n',
        encoding="utf-8",
    )

    records = _read_records(path)

    assert records == [{"input_ids": [1, 2], "labels": [-100, 2]}]


def test_select_losses_prefers_the_aggregate_train_loss() -> None:
    history = [{"loss": 9.0}, {"loss": 5.0}, {"loss": 2.0}]

    train_loss, final_loss = _select_losses({"train_loss": 3.5}, history)

    assert train_loss == 3.5
    assert final_loss == 2.0


def test_select_losses_falls_back_to_the_last_logged_loss_not_the_first() -> None:
    history = [{"loss": 9.0}, {"loss": 5.0}, {"loss": 2.0}]

    train_loss, final_loss = _select_losses({}, history)

    # Using the first logged step would report 9.0 and understate convergence.
    assert train_loss == 2.0
    assert final_loss == 2.0


def test_select_losses_handles_empty_history() -> None:
    assert _select_losses({}, []) == (0.0, 0.0)
