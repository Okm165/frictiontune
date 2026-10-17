"""Tests for the IO on both sides of the model boundary.

A corrupt base-model config or a half-written SFT artifact must fail with a
named stage error, not a bare ``json.JSONDecodeError`` traceback, so a real run
diagnoses the cause instead of the symptom.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from frictiontune.config.models import HardwareConfig
from frictiontune.core.errors import StageError
from frictiontune.stages.fetch_base import probe_base
from frictiontune.stages.train_sft import TrainSftSettings, _epochs_for
from frictiontune_train.sft import _PadCollator, _read_records, _select_losses

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


def test_pad_collator_windows_records_to_the_declared_max_length() -> None:
    pytest.importorskip("torch")
    collator = _PadCollator(pad_token_id=0, max_length=3)

    batch = collator(
        [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [7], "labels": [7]},
        ]
    )

    # The longest record keeps its tail (the assistant behaviour lives there);
    # shorter rows pad with the pad token and the ignore index.
    assert batch["input_ids"].tolist() == [[3, 4, 5], [7, 0, 0]]
    assert batch["labels"].tolist() == [[3, 4, 5], [7, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def _hardware(devices: int = 3, accum: int = 4) -> HardwareConfig:
    return HardwareConfig(devices=devices, grad_accum=accum, per_device_batch=1)


def test_epochs_are_raised_when_the_curated_set_is_tiny() -> None:
    # The real shape: 23 records, 3 ranks x batch 1 x accumulation 4 = 12
    # effective, so 3 epochs is 6 optimizer steps. The budget is steps, not
    # epochs, so the epoch count has to rise to reach it.
    settings = TrainSftSettings(min_steps=60)

    assert _epochs_for(23, _hardware(), settings) == 30.0


def test_epochs_are_left_alone_when_the_set_is_large_enough() -> None:
    settings = TrainSftSettings(min_steps=60)

    # 12 000 records / 12 effective = 1000 steps per epoch, so 3 epochs is
    # already far past the budget and must not be touched.
    assert _epochs_for(12_000, _hardware(), settings) == 3.0


def test_epochs_are_capped_so_a_tiny_set_cannot_loop_forever() -> None:
    settings = TrainSftSettings(min_steps=10_000, max_epochs=25.0)

    assert _epochs_for(1, _hardware(), settings) == 25.0


def test_epochs_are_derived_from_the_topology_not_a_constant() -> None:
    settings = TrainSftSettings(min_steps=60)

    # 200 records: one card with no accumulation clears the budget inside 3
    # epochs (200 steps each), while three cards x accumulation 4 need ~4
    # (17 steps each). The epoch count follows the topology, not a constant.
    assert _epochs_for(200, _hardware(devices=1, accum=1), settings) == 3.0
    assert _epochs_for(200, _hardware(), settings) > 3.0


def test_a_zero_step_budget_is_a_no_op() -> None:
    settings = TrainSftSettings(min_steps=0)

    assert _epochs_for(23, _hardware(), settings) == 3.0
