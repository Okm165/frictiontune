"""DPO preference training over the pairs mined from friction recoveries.

The second half of GAP-104: :mod:`frictiontune.stages._extensions` mines
chosen/rejected pairs — the assistant's verbatim failing turns against the same
session's recovery, under the same pre-friction prompt — and this module turns
them into an avoidance policy with TRL's DPO trainer. DPO (Rafailov et al.,
2023) needs no reward model and no rollouts: the objective pushes the
likelihoods of the two continuations apart under the same prompt, which is
exactly the thing positive-only SFT structurally cannot express.

The job *continues* from the SFT adapter (the standard SFT-then-DPO ordering).
The adapter's LoRA configuration is loaded from disk. The reference policy is
a frozen clone of that same SFT adapter — NOT the adapter-disabled base that
TRL's ``ref_model=None`` shortcut would use: the SFT run trained on the chosen
side, so measured against the base the preference margin starts pre-saturated
and the objective looks already solved.

The Unsloth fast path is deliberately not used here: it is a single-card loader
with its own trainer assumptions, and DPO runs under the same data-parallel
topology as SFT (one card, or one rank of a torchrun group).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frictiontune.backends.models import TrainingResult
from frictiontune.domain.trace import Message

from frictiontune_train import distributed as _dist
from frictiontune_train.lazy import require, torch_hint
from frictiontune_train.loaders import load_tokenizer, load_transformers_base, save_adapter

if TYPE_CHECKING:
    from frictiontune.backends.models import DpoRequest


def to_chat_message_dicts(messages: list[Any]) -> list[dict[str, str]]:
    """Render canonical messages the same way SFT renders them, as plain dicts.

    Sharing the SFT renderer is the point: the preference objective must see
    the conversation exactly as the supervised stage taught it, or the two
    objectives would optimize different views of the same behaviour.
    """
    from frictiontune.sft.renderer import to_chat_messages

    return [
        {"role": message.role, "content": message.content}
        for message in to_chat_messages(messages, include_reasoning=False)
    ]


def load_pairs(pairs_dir: str) -> list[dict[str, Any]]:
    """Read the mined pairs and convert them to TRL's conversational format."""
    path = Path(pairs_dir) / "preference_pairs.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing preference pairs file {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        rows.append(
            {
                "prompt": to_chat_message_dicts(
                    [Message.model_validate(m) for m in pair.get("prompt") or []]
                ),
                "chosen": to_chat_message_dicts(
                    [Message.model_validate(m) for m in pair.get("chosen") or []]
                ),
                "rejected": to_chat_message_dicts(
                    [Message.model_validate(m) for m in pair.get("rejected") or []]
                ),
            }
        )
    if not rows:
        raise ValueError("preference dataset is empty")
    return rows


def run_dpo(request: DpoRequest) -> TrainingResult:
    """Execute one DPO job and return its artifacts."""
    torch = require("torch", hint=torch_hint())
    datasets = require("datasets", hint=torch_hint())
    trl = require("trl", hint=torch_hint())
    peft = require("peft", hint=torch_hint())

    rows = load_pairs(request.pairs_dir)

    if _dist.is_distributed():
        torch.cuda.set_device(_dist.local_rank())
    tokenizer = load_tokenizer(request.base_model)
    base = load_transformers_base(
        request.base_model, request.hardware, device_map=_dist.device_map()
    )
    model = peft.PeftModel.from_pretrained(base, request.adapter_dir, is_trainable=True)
    # The reference policy is the *SFT model*, not the bare base — the
    # canonical DPO setup (Rafailov et al. 2023; "pi_ref typically corresponds
    # to the policy obtained after SFT", arXiv:2506.04272). With ref_model=None
    # and a PEFT model, TRL uses the adapter-disabled pass — the bare base —
    # and that would be wrong here twice over: the SFT adapter was trained ON
    # the chosen side, so the initial margin against the base is already
    # substantially positive and the gradient near zero (the objective would
    # look already solved). A frozen clone of the SFT adapter makes the margin
    # start at exactly zero. The clone costs a second base copy;
    # precompute_ref_log_probs pays it once instead of per step.
    ref_base = load_transformers_base(
        request.base_model, request.hardware, device_map=_dist.device_map()
    )
    ref_model = peft.PeftModel.from_pretrained(ref_base, request.adapter_dir)
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    max_prompt = request.max_prompt_length or (request.max_seq_len * 3) // 4
    max_completion = request.max_completion_length or request.max_seq_len // 4
    config = trl.DPOConfig(
        output_dir=request.output_dir,
        per_device_train_batch_size=request.hardware.per_device_batch,
        gradient_accumulation_steps=request.hardware.grad_accum,
        learning_rate=request.learning_rate,
        num_train_epochs=request.epochs,
        max_steps=request.max_steps,
        beta=request.beta,
        max_length=request.max_seq_len,
        max_prompt_length=max_prompt,
        max_completion_length=max_completion,
        precompute_ref_log_probs=True,
        logging_steps=10,
        save_total_limit=2,
        bf16=request.hardware.dtype == "bf16",
        fp16=request.hardware.dtype == "fp16",
        gradient_checkpointing=request.hardware.gradient_checkpointing,
        seed=request.seed,
        report_to=[],
        dataloader_num_workers=int(os.environ.get("FRICTIONTUNE_TRAIN_DATALOADER_WORKERS", "2")),
    )
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=tokenizer,
    )

    started = time.time()
    train_output = trainer.train()
    elapsed = time.time() - started

    adapter_dir = request.output_dir
    checkpoints: list[str] = []
    metrics = {
        str(key): float(value)
        for key, value in (train_output.metrics or {}).items()
        if isinstance(value, (int, float))
    }
    if _dist.is_main_process():
        adapter_dir = save_adapter(model, request.output_dir, tokenizer)
        checkpoints = sorted(str(path) for path in Path(adapter_dir).glob("checkpoint-*"))
    return TrainingResult(
        adapter_dir=adapter_dir,
        steps=int(trainer.state.global_step),
        train_loss=float(metrics.get("train_loss", 0.0)),
        final_loss=float(metrics.get("train_loss", 0.0)),
        wall_time_s=elapsed,
        metrics=metrics,
        checkpoints=checkpoints,
    )
