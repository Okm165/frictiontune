"""Tests for the DPO preference-training path (GAP-104, backend half).

Only the deterministic plumbing is unit-tested here: the request contract, the
pair-to-dataset conversion, and the dispatch wiring. The trainer itself needs a
GPU and is exercised by a gated run.
"""

from __future__ import annotations

from pathlib import Path

from frictiontune.backends.models import DpoRequest
from frictiontune.config.models import HardwareConfig
from frictiontune.domain.dataset import PreferencePair
from frictiontune.domain.trace import Message, TextPart
from frictiontune_train.dpo import load_pairs, to_chat_message_dicts


def _message(identifier: str, role: str, text: str) -> Message:
    return Message(
        id=identifier,
        session_id="ses_x",
        seq=0,
        role=role,
        time_created=0,
        parts=[TextPart(text=text)],
    )


def _request(tmp_path: Path) -> DpoRequest:
    tmp = Path(tmp_path)
    return DpoRequest(
        base_model="org/model",
        adapter_dir=str(tmp / "sft-adapter"),
        pairs_dir=str(tmp),
        output_dir=str(tmp / "out"),
        hardware=HardwareConfig(),
    )


def test_dpo_request_continues_an_adapter_and_carries_no_lora_shape(tmp_path: Path) -> None:
    request = _request(tmp_path)
    # The LoRA shape is the SFT run's decision, loaded from the adapter itself.
    assert not hasattr(request, "lora_r")
    assert request.beta == 0.1
    assert request.learning_rate < 5e-5  # an order below the SFT default 2e-4


def test_pairs_convert_to_trl_conversational_format(tmp_path: Path) -> None:
    pair = PreferencePair(
        id="ex1/pair",
        source="ses_x:ep1",
        prompt=[_message("m1", "user", "run the tests")],
        chosen=[_message("m2", "assistant", "here is the failing output")],
        rejected=[_message("m3", "assistant", "all tests pass")],
        meta={"event_id": "ev1"},
    )
    (tmp_path / "preference_pairs.jsonl").write_text(
        pair.model_dump_json() + "\n", encoding="utf-8"
    )

    rows = load_pairs(str(tmp_path))

    assert len(rows) == 1
    assert rows[0]["prompt"] == [{"role": "user", "content": "run the tests"}]
    assert rows[0]["chosen"][0]["role"] == "assistant"
    assert "failing output" in rows[0]["chosen"][0]["content"]
    assert rows[0]["rejected"][0]["content"] == "all tests pass"


def test_load_pairs_fails_loudly_on_a_missing_or_empty_file(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_pairs(str(tmp_path))
    (tmp_path / "preference_pairs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_pairs(str(tmp_path))


def test_rendering_matches_the_sft_view() -> None:
    # The conversion must go through the same renderer SFT trained with.
    message = _message("m1", "user", "hello")
    assert to_chat_message_dicts([message]) == [{"role": "user", "content": "hello"}]


def test_train_dpo_spec_wiring() -> None:
    from frictiontune.pipeline.dag import full_graph
    from frictiontune.stages._extensions import TrainDpo

    graph = full_graph()
    assert graph.predecessors("train_dpo") == ("build_preference_pairs",)
    spec = TrainDpo.spec
    assert spec.tier.value == "C"
    assert not spec.requires_llm
    assert "preference_pairs" in spec.inputs
