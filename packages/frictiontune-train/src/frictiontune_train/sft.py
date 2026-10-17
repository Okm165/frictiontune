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
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frictiontune.backends.models import TrainingRequest, TrainingResult
from frictiontune.core.errors import StageError

from frictiontune_train import distributed
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


def _early_stopping(has_eval: bool, patience: int = 3) -> list[Any]:
    """Stop when the held-out loss has not improved for ``patience`` epochs.

    The budget a small curated set needs is not knowable in advance, so the run
    is given a generous ceiling and the held-out loss decides where to stop. On
    the real corpus that ceiling was 30 epochs and the minimum fell at epoch 8
    (eval loss 1.927, rising to 4.400 by epoch 30): the best-checkpoint
    selection still shipped the right adapter, but two thirds of the GPU time
    went into epochs that could only be discarded. Patience bounds that waste
    without deciding the epoch in advance.

    Without a held-out split there is nothing to stop on, so no callback is
    registered.
    """
    if not has_eval:
        return []
    from transformers import EarlyStoppingCallback

    return [EarlyStoppingCallback(early_stopping_patience=patience)]


def _load_eval(request: TrainingRequest, datasets: Any) -> Any | None:
    """Load the held-out split, or return ``None`` when there is none.

    The validation file is written by ``decontaminate`` next to ``train.jsonl``
    (``val.jsonl``; ``eval.jsonl`` is accepted for artifacts produced before the
    three-way split). A run without one, or with an empty one, simply trains
    without evaluation instead of failing: the selection boundary is a data
    decision, not a trainer one. This set is used for early stopping and
    best-checkpoint selection only — never to report final performance.
    """
    if not request.eval_dir:
        return None
    directory = Path(request.eval_dir)
    path = directory / "val.jsonl"
    if not path.is_file():
        path = directory / "eval.jsonl"
    if not path.is_file():
        return None
    records = _read_records(path)
    return datasets.Dataset.from_list(records) if records else None


def _select_losses(
    metrics: dict[str, float], history: Sequence[dict[str, Any]]
) -> tuple[float, float]:
    """Return ``(train_loss, final_loss)`` from trainer metrics and log history.

    The trainer's aggregate ``train_loss`` is the mean over the whole run and is
    preferred. When it is absent, fall back to the *last* logged loss, never the
    first: the first logged step is not the training loss and reading it as such
    understates convergence. Short runs log no per-step loss at all, so
    ``final_loss`` then falls back to the aggregate rather than to a zero that
    would read as a perfect fit.
    """
    train_loss = float(metrics.get("train_loss", history[-1]["loss"] if history else 0.0))
    final_loss = float(history[-1]["loss"]) if history else train_loss
    return train_loss, final_loss


def _collator(pad_token_id: int, max_length: int) -> Any:
    """Build a padding collator that pads labels with ``-100``."""
    return _PadCollator(pad_token_id, max_length)


class _PadCollator:
    """Pad a batch to its longest row; labels pad with ``-100``.

    Records are windowed to ``max_length`` when they are longer. The tail is
    kept, not the head: the masked SFT records put the assistant behaviour the
    model is being taught at the end of the sequence, so dropping the earliest
    context is the side that preserves the learning signal. Without this the
    batch would silently grow to a record's raw length and the trainer would OOM
    on a window it declared it could not afford.

    Module scope on purpose: with ``dataloader_num_workers > 0`` the collator is
    pickled to the workers, and a class defined inside a function is not
    picklable.
    """

    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self._pad_token_id = pad_token_id
        self._max_length = max_length

    def _window(self, tokens: Sequence[int]) -> list[int]:
        """Keep the last ``max_length`` tokens, or the whole record if shorter."""
        if self._max_length > 0 and len(tokens) > self._max_length:
            return list(tokens[-self._max_length :])
        return list(tokens)

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        torch = require("torch", hint=torch_hint())
        windowed = [(self._window(f["input_ids"]), self._window(f["labels"])) for f in features]
        max_len = max(len(tokens) for tokens, _ in windowed)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        for tokens, row_labels in windowed:
            pad = max_len - len(tokens)
            input_ids.append(list(tokens) + [self._pad_token_id] * pad)
            labels.append(list(row_labels) + [_IGNORE_INDEX] * pad)
            attention.append([1] * len(tokens) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def load_model_and_tokenizer(
    request: TrainingRequest, *, device_map: Any = "auto"
) -> tuple[Any, Any]:
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
    model = load_transformers_base(request.base_model, request.hardware, device_map=device_map)
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


def _familiarity_keep_set(
    losses: dict[str, float], known_fraction: float
) -> tuple[set[str], dict[str, str]]:
    """Pick the records to keep under the Gekhman arm (GAP-096).

    ``losses`` maps record id to mean label-loss at base init. Keeping the
    *lowest-loss* fraction keeps the examples the model already knows how to
    produce — the ones that teach efficient use of existing knowledge instead
    of new facts. Returns the keep set and the drop reason per dropped id, so
    the stage's report can show exactly who was excluded and why.
    """
    if known_fraction >= 1.0 or not losses:
        return set(losses), {}
    keep_count = max(1, round(len(losses) * known_fraction))
    ranked = sorted(losses.items(), key=lambda item: (item[1], item[0]))
    dropped = ranked[keep_count:]
    keep = {identifier for identifier, _loss in ranked[:keep_count]}
    return keep, {identifier: "unknown_at_init" for identifier, _loss in dropped}


def run_sft(request: TrainingRequest) -> TrainingResult:
    """Execute one supervised QLoRA job and return its artifacts."""
    torch = require("torch", hint=torch_hint())
    datasets = require("datasets", hint=torch_hint())
    trl = require("trl", hint=torch_hint())

    train_path = Path(request.dataset_dir) / "train.jsonl"
    if not train_path.is_file():
        raise StageError("train_sft", f"missing training file {train_path}")
    records = _read_records(train_path)
    if not records:
        raise StageError("train_sft", "training dataset is empty")

    if distributed.is_distributed():
        # Bind this process to its row of the device group before any allocation.
        torch.cuda.set_device(distributed.local_rank())
    model, tokenizer = load_model_and_tokenizer(request, device_map=distributed.device_map())
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # The Gekhman arm: when a fraction below 1.0 is requested, measure each
    # record's mean label-loss at base init (one forward pass, no gradient) and
    # drop the highest-loss records — the facts the model does not already
    # know. The familiarity table lands next to the adapter as evidence, and
    # the drop is mirrored in the result so the stage's report can show it.
    familiarity: dict[str, float] = {}
    if request.known_fraction < 1.0:
        model.eval()
        losses: dict[str, float] = {}
        with torch.no_grad():
            for index, record in enumerate(records):
                row = _collator(pad_token_id, request.max_seq_len)([record])
                outputs = model(**row)
                labels = row["labels"]
                masked = labels != -100
                if masked.any():
                    losses[str(record.get("id", index))] = float(
                        outputs.loss.item()
                        if outputs.loss.dim() == 0
                        else outputs.loss[masked].mean().item()
                    )
        familiarity = losses
        keep, _dropped_reasons = _familiarity_keep_set(losses, request.known_fraction)
        records = [record for record in records if str(record.get("id")) in keep]
        model.train()
    familiarity_path = Path(request.output_dir) / "familiarity.json"
    if familiarity:
        familiarity_path.parent.mkdir(parents=True, exist_ok=True)
        familiarity_path.write_text(
            json.dumps(
                {
                    "known_fraction": request.known_fraction,
                    "losses": familiarity,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    train_dataset = datasets.Dataset.from_list(records)
    eval_dataset = _load_eval(request, datasets)

    settings: dict[str, Any] = {
        "output_dir": request.output_dir,
        "per_device_train_batch_size": request.hardware.per_device_batch,
        # Evaluation must use the same batch size the hardware preset has proven
        # fits for training. Left unset, the trainer falls back to its own default
        # of 8, and a held-out split of full-length records then asks for eight
        # times the activation memory of a training step — the one allocation that
        # reliably OOMs a single 24 GB card even when training itself fits.
        "per_device_eval_batch_size": request.hardware.per_device_batch,
        "gradient_accumulation_steps": request.hardware.grad_accum,
        "learning_rate": request.learning_rate,
        "num_train_epochs": request.epochs,
        "max_steps": request.max_steps,
        "max_seq_length": request.max_seq_len,
        "logging_steps": 10,
        "save_total_limit": 2,
        "bf16": request.hardware.dtype == "bf16",
        "fp16": request.hardware.dtype == "fp16",
        "gradient_checkpointing": request.hardware.gradient_checkpointing,
        "seed": request.seed,
        "report_to": [],
        # Feed the GPUs from worker processes so the whole CPU participates in
        # training instead of the main process serialising every batch.
        "dataloader_num_workers": int(os.environ.get("FRICTIONTUNE_TRAIN_DATALOADER_WORKERS", "2")),
        # The dataset is already tokenized and masked; do not re-tokenize.
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        # The held-out loss is the only eval output we record; there is no
        # ``compute_metrics``. Telling the trainer that also keeps the epoch-end
        # evaluation on the memory-efficient fused head instead of building the
        # vocabulary-wide logits a second time, which is what pushed a 24 GB card
        # over the edge.
        "prediction_loss_only": True,
        **distributed.config_kwargs(),
    }
    if eval_dataset is not None:
        # Evaluate every epoch and keep the checkpoint with the *best* held-out
        # loss, not merely the last. Over a handful of epochs on a few dozen
        # records the final step is usually the most over-fit one, so selecting
        # on held-out loss is the difference between shipping the best adapter
        # and shipping the noisiest. transformers requires the save and eval
        # strategies to match for the selection to work.
        settings.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        # No held-out split: keep periodic step checkpoints instead.
        settings["save_strategy"] = "steps"
        settings["save_steps"] = request.save_steps
    config_cls = getattr(trl, "SFTConfig", None) or trl.TrainingArguments
    config = config_cls(
        **resolve_config_kwargs(
            config_cls,
            settings,
            # ``max_seq_length`` is the pre-0.17 name; current TRL uses
            # ``max_length`` and removes the old spelling in 2.0.
            # ``eval_strategy`` was ``evaluation_strategy`` before transformers 5.
            aliases={
                "max_seq_length": ("max_length", "max_seq_length"),
                "eval_strategy": ("eval_strategy", "evaluation_strategy"),
            },
            stage="train_sft",
        )
    )
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_collator(pad_token_id, request.max_seq_len),
        args=config,
        **trainer_token_kwargs(trl.SFTTrainer, tokenizer, stage="train_sft"),
        callbacks=_early_stopping(eval_dataset is not None),
    )

    started = time.time()
    train_output = trainer.train(resume_from_checkpoint=request.resume_from)
    elapsed = time.time() - started

    # Only rank 0 owns the adapter; the other ranks hold identical replicas and
    # must not race on the output directory.
    adapter_dir = request.output_dir
    checkpoints: list[str] = []
    if distributed.is_main_process():
        adapter_dir = save_adapter(model, request.output_dir, tokenizer)
        checkpoints = sorted(str(path) for path in Path(adapter_dir).glob("checkpoint-*"))
    metrics = {
        str(key): float(value)
        for key, value in (train_output.metrics or {}).items()
        if isinstance(value, (int, float))
    }
    # The held-out loss is what the epoch-end evaluation buys; it lands in the
    # trainer's log history rather than in ``train_output.metrics`` when the run
    # stops before the epoch boundary, so fold it in explicitly rather than
    # computing it and dropping it.
    for entry in trainer.state.log_history:
        for key, value in entry.items():
            if key.startswith("eval_") and isinstance(value, (int, float)):
                metrics[key] = float(value)
    # The best held-out loss is what `load_best_model_at_end` selected on; it is
    # the y-axis of any learning curve drawn over training subsets, so it is
    # surfaced explicitly rather than left for callers to re-derive from the
    # per-epoch history.
    if trainer.state.best_metric is not None:
        metrics["best_eval_loss"] = float(trainer.state.best_metric)
    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    train_loss, final_loss = _select_losses(metrics, history)
    return TrainingResult(
        adapter_dir=adapter_dir,
        steps=int(trainer.state.global_step),
        train_loss=train_loss,
        final_loss=final_loss,
        wall_time_s=elapsed,
        metrics=metrics,
        checkpoints=checkpoints,
    )
