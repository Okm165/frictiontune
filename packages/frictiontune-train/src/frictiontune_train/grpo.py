"""GRPO training with a judged reward.

This runs after SFT has plateaued and only when the reward has been calibrated:
a judge endpoint is required so the reward is not a constant. It reuses the same
loader as SFT, continues from the SFT adapter when one is given, and optimises
the reward defined in :mod:`frictiontune_train.reward`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from frictiontune.backends.models import GrpoRequest, TrainingRequest, TrainingResult
from frictiontune.core.errors import StageError

from frictiontune_train.lazy import require, torch_hint
from frictiontune_train.loaders import continue_from_adapter, load_tokenizer, save_adapter
from frictiontune_train.reward import build_reward
from frictiontune_train.sft import load_model_and_tokenizer
from frictiontune_train.trl_compat import resolve_config_kwargs, trainer_token_kwargs


def _read_prompts(path: Path) -> list[str]:
    """Read a JSONL prompts file written by the ``train_grpo`` stage."""
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageError("train_grpo", f"{path}: malformed JSON line: {exc}") from exc
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
    if not prompts:
        raise StageError("train_grpo", f"no prompts found in {path}")
    return prompts


def _load_for_grpo(request: GrpoRequest) -> tuple[Any, Any]:
    """Load a model to continue RL from: the SFT adapter if present, else base."""
    if request.adapter_dir and Path(request.adapter_dir).is_dir():
        # Continue training an existing adapter rather than re-initialising LoRA.
        tokenizer = load_tokenizer(request.base_model)
        model = continue_from_adapter(request.base_model, request.adapter_dir, request.hardware)
        return model, tokenizer

    request_stub = TrainingRequest(
        base_model=request.base_model,
        dataset_dir=".",
        output_dir=request.output_dir,
        hardware=request.hardware,
        seed=request.seed,
        max_seq_len=request.hardware.max_seq_len,
    )
    return load_model_and_tokenizer(request_stub)


def run_grpo(request: GrpoRequest) -> TrainingResult:
    """Execute one GRPO job and return its artifacts."""
    require("torch", hint=torch_hint())
    datasets = require("datasets", hint=torch_hint())
    trl = require("trl", hint=torch_hint())

    prompts_path = Path(request.prompts_file)
    if not prompts_path.is_file():
        raise StageError("train_grpo", f"missing prompts file {prompts_path}")
    prompts = _read_prompts(prompts_path)

    model, tokenizer = _load_for_grpo(request)
    dataset = datasets.Dataset.from_dict({"prompt": prompts})
    reward = build_reward(
        request.reward_weights,
        judge_url=os.environ.get("FRICTIONTUNE_GRPO_JUDGE_URL"),
    )

    config_cls = getattr(trl, "GRPOConfig", None) or trl.TrainingArguments
    config = config_cls(
        **resolve_config_kwargs(
            config_cls,
            {
                "output_dir": request.output_dir,
                "learning_rate": request.learning_rate,
                "max_steps": request.max_steps,
                "per_device_train_batch_size": request.hardware.per_device_batch,
                "gradient_accumulation_steps": request.hardware.grad_accum,
                "num_generations": request.group_size,
                "max_new_tokens": request.max_new_tokens,
                "bf16": request.hardware.dtype == "bf16",
                "fp16": request.hardware.dtype == "fp16",
                "gradient_checkpointing": request.hardware.gradient_checkpointing,
                "seed": request.seed,
                "report_to": [],
            },
            # GRPO generates via ``max_completion_length``; ``max_new_tokens``
            # is a ``generate`` kwarg that GRPOConfig would silently ignore.
            aliases={"max_new_tokens": ("max_completion_length", "max_new_tokens")},
            stage="train_grpo",
        )
    )
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
        **trainer_token_kwargs(trl.GRPOTrainer, tokenizer, stage="train_grpo"),
    )

    started = time.time()
    train_output = trainer.train()
    elapsed = time.time() - started

    adapter_dir = save_adapter(model, request.output_dir, tokenizer)
    metrics = {
        str(key): float(value)
        for key, value in (train_output.metrics or {}).items()
        if isinstance(value, (int, float))
    }
    history = [entry for entry in trainer.state.log_history if "reward" in entry]
    reward_value = float(history[-1]["reward"]) if history else metrics.get("reward", 0.0)
    metrics.setdefault("reward", reward_value)
    return TrainingResult(
        adapter_dir=adapter_dir,
        steps=int(trainer.state.global_step),
        train_loss=metrics.get("loss", 0.0),
        final_loss=metrics.get("loss", 0.0),
        wall_time_s=elapsed,
        metrics=metrics,
        checkpoints=sorted(str(path) for path in Path(adapter_dir).glob("checkpoint-*")),
    )
