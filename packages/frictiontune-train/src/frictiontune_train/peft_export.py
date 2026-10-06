"""Adapter export: merge/pack for serving, then verify parity.

Export is where a silent regression can slip in (a bad merge produces a model
that loads but answers differently). The parity probe compares greedy outputs
from the base model and the exported model on a small prompt set, so any
divergence is recorded in the run artifacts rather than discovered in
production.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from frictiontune.backends.models import ExportRequest, ExportResult, ParityResult
from frictiontune.config.models import HardwareConfig

from frictiontune_train.lazy import require, torch_hint
from frictiontune_train.loaders import load_tokenizer, load_transformers_base


def full_precision_hardware() -> HardwareConfig:
    """Hardware settings for loading a base model to merge in full precision."""
    return HardwareConfig(quant="none", offload="none", dtype="bf16")


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64) -> str:
    """Greedy-generate a single completion for a prompt."""
    torch = require("torch", hint=torch_hint())
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


class LocalServingBackend:
    """Merges or packages an adapter locally and reports parity."""

    async def export(self, request: ExportRequest) -> ExportResult:
        """Export the adapter (blocking work runs in a thread)."""
        import asyncio

        return await asyncio.to_thread(self._export_sync, request)

    def _export_sync(self, request: ExportRequest) -> ExportResult:
        served_dir = Path(request.output_dir)
        served_dir.mkdir(parents=True, exist_ok=True)
        if request.method == "merge":
            return self._merge(request, served_dir)
        self._package_lora(request, served_dir)
        # Parity is only meaningful for a merged model we can load again.
        return ExportResult(
            served_dir=str(served_dir),
            merged=False,
            parity=ParityResult(total=0, matches=0, mismatches=[]),
        )

    def _merge(self, request: ExportRequest, served_dir: Path) -> ExportResult:
        peft = require("peft", hint=torch_hint())
        tokenizer = load_tokenizer(request.base_model)
        base = load_transformers_base(request.base_model, full_precision_hardware())
        base_outputs = (
            [_generate(base, tokenizer, probe.prompt) for probe in request.probe]
            if request.probe
            else []
        )
        loaded = peft.PeftModel.from_pretrained(base, request.adapter_dir)
        model = loaded.merge_and_unload()
        model.save_pretrained(str(served_dir))
        tokenizer.save_pretrained(str(served_dir))

        mismatches: list[str] = []
        matches = 0
        for probe, reference in zip(request.probe, base_outputs, strict=False):
            expected = probe.expected if probe.expected is not None else reference
            if _generate(model, tokenizer, probe.prompt) == expected:
                matches += 1
            else:
                mismatches.append(probe.prompt)
        return ExportResult(
            served_dir=str(served_dir),
            merged=True,
            parity=ParityResult(total=len(request.probe), matches=matches, mismatches=mismatches),
        )

    def _package_lora(self, request: ExportRequest, served_dir: Path) -> None:
        """Copy the LoRA next to a config so ``vllm --enable-lora`` can serve it."""
        for item in Path(request.adapter_dir).glob("*"):
            target = served_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
