"""Supervised QLoRA fine-tuning on frictiontune's masked SFT records.

The dataset handed to this module is the ``train.jsonl`` produced by
``decontaminate``: each row already carries ``input_ids`` and ``labels`` with
``-100`` on everything that is context or environment output. Training therefore
happens on the assistant behaviour only, exactly as the analysis decided.

Design notes:

* **MoE trap.** ``target_parameters`` defaults to an empty list, so LoRA never
  attaches to a model's fused expert weights. bitsandbytes also silently skips
  fused-expert ``Linear4bit`` modules, so the combination is a known failure mode
  we avoid by construction.
* **Unsloth is optional.** It is used when importable and enabled; otherwise a
  plain transformers + peft path runs. Both share the same argument assembly.
* **Everything runs in a worker thread** (see :mod:`frictiontune_train.training`);
  this module is synchronous on purpose.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frictiontune.backends.models import TrainingRequest, TrainingResult
from frictiontune.core.errors import StageError

from frictiontune_train.lazy import require, torch_hint
from frictiontune_train.loaders import (
    apply_lora,
    load_tokenizer,
    load_transformers_base,
    load_unsloth,
    save_adapter,
)
from frictiontune_train.trl_compat import resolve_config_kwargs, trainer_token_kwargs

if TYPE_CHECKING:
    from collections.abc import Sequence

_IGNORE_INDEX = -100


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL SFT records, keeping only the token and label columns."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageError("train_sft", f"{path}: malformed JSON line: {exc}") from exc
            if not isinstance(row, dict) or "input_ids" not in row or "labels" not in row:
                identifier = row.get("id") if isinstance(row, dict) else None
                raise StageError("train_sft", f"record {identifier!r} has no input_ids/labels")
            records.append({"input_ids": row["input_ids"], "labels": row["labels"]})
    return records


def _select_losses(
    metrics: dict[str, float], history: Sequence[dict[str, Any]]
) -> tuple[float, float]:
    """Return ``(train_loss, final_loss)`` from trainer metrics and log history.

    The trainer's aggregate ``train_loss`` is the mean over the whole run and is
    preferred. When it is absent, fall back to the *last* logged loss, never the
    first: the first logged step is not the training loss and reading it as such
    understates convergence.
    """
    final_loss = float(history[-1]["loss"]) if history else 0.0
    train_loss = float(metrics.get("train_loss", final_loss))
    return train_loss, final_loss


def _collator(pad_token_id: int) -> Any:
    """Build a padding collator that pads labels with ``-100``."""
    torch = require("torch", hint=torch_hint())

    class _Collator:
        def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids: list[list[int]] = []
            labels: list[list[int]] = []
            attention: list[list[int]] = []
            for feature in features:
                tokens = feature["input_ids"]
                pad = max_len - len(tokens)
                input_ids.append(tokens + [pad_token_id] * pad)
                labels.append(feature["labels"] + [_IGNORE_INDEX] * pad)
                attention.append([1] * len(tokens) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }

    return _Collator()


def load_model_and_tokenizer(request: TrainingRequest) -> tuple[Any, Any]:
    """Load the base model and attach LoRA, preferring Unsloth when available."""
    unsloth = load_unsloth(
        base_model=request.base_model,
        max_seq_len=request.max_seq_len,
        quant=request.hardware.quant,
    )
    if unsloth is not None:
        model, tokenizer = unsloth
        fast_model = require("unsloth", hint="pip install unsloth (optional)")
        model = fast_model.FastLanguageModel.get_peft_model(
            model,
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            target_modules=request.target_modules,
            use_gradient_checkpointing=request.hardware.gradient_checkpointing,
            random_state=request.seed,
        )
        return model, tokenizer

    tokenizer = load_tokenizer(request.base_model)
    model = load_transformers_base(request.base_model, request.hardware)
    model = apply_lora(
        model,
        hardware=request.hardware,
        lora_r=request.lora_r,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
        target_modules=request.target_modules,
        target_parameters=request.target_parameters,
    )
    return model, tokenizer


def run_sft(request: TrainingRequest) -> TrainingResult:
    """Execute one supervised QLoRA job and return its artifacts."""
    require("torch", hint=torch_hint())
    datasets = require("datasets", hint=torch_hint())
    trl = require("trl", hint=torch_hint())

    train_path = Path(request.dataset_dir) / "train.jsonl"
    if not train_path.is_file():
        raise StageError("train_sft", f"missing training file {train_path}")
    records = _read_records(train_path)
    if not records:
        raise StageError("train_sft", "training dataset is empty")

    model, tokenizer = load_model_and_tokenizer(request)
    train_dataset = datasets.Dataset.from_list(records)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    config_cls = getattr(trl, "SFTConfig", None) or trl.TrainingArguments
    config = config_cls(
        **resolve_config_kwargs(
            config_cls,
            {
                "output_dir": request.output_dir,
                "per_device_train_batch_size": request.hardware.per_device_batch,
                "gradient_accumulation_steps": request.hardware.grad_accum,
                "learning_rate": request.learning_rate,
                "num_train_epochs": request.epochs,
                "max_steps": request.max_steps,
                "max_seq_length": request.max_seq_len,
                "logging_steps": 10,
                "save_steps": request.save_steps,
                "save_total_limit": 2,
                "bf16": request.hardware.dtype == "bf16",
                "fp16": request.hardware.dtype == "fp16",
                "gradient_checkpointing": request.hardware.gradient_checkpointing,
                "seed": request.seed,
                "report_to": [],
                # The dataset is already tokenized and masked; do not re-tokenize.
                "dataset_kwargs": {"skip_prepare_dataset": True},
                "remove_unused_columns": False,
            },
            # ``max_seq_length`` is the pre-0.17 name; current TRL uses
            # ``max_length`` and removes the old spelling in 2.0.
            aliases={"max_seq_length": ("max_length", "max_seq_length")},
            stage="train_sft",
        )
    )
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=_collator(pad_token_id),
        args=config,
        **trainer_token_kwargs(trl.SFTTrainer, tokenizer, stage="train_sft"),
    )

    started = time.time()
    train_output = trainer.train(resume_from_checkpoint=request.resume_from)
    elapsed = time.time() - started

    adapter_dir = save_adapter(model, request.output_dir, tokenizer)
    metrics = {
        str(key): float(value)
        for key, value in (train_output.metrics or {}).items()
        if isinstance(value, (int, float))
    }
    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    train_loss, final_loss = _select_losses(metrics, history)
    checkpoints = sorted(str(path) for path in Path(adapter_dir).glob("checkpoint-*"))
    return TrainingResult(
        adapter_dir=adapter_dir,
        steps=int(trainer.state.global_step),
        train_loss=train_loss,
        final_loss=final_loss,
        wall_time_s=elapsed,
        metrics=metrics,
        checkpoints=checkpoints,
    )
