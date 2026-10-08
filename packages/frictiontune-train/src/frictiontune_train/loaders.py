"""Single definition of model/tokenizer loading for SFT and GRPO.

Both training paths must load the base model, apply the same quantisation, and
attach LoRA the same way; duplicating that logic is how MoE and dtype bugs creep
in. This module owns it once.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from frictiontune_train.lazy import require, require_attr, torch_hint

if TYPE_CHECKING:
    from frictiontune.config.models import HardwareConfig


def compute_dtype(hardware: HardwareConfig, torch: Any) -> Any:
    """Return the torch dtype named by the hardware preset."""
    return torch.bfloat16 if hardware.dtype == "bf16" else torch.float16


def load_tokenizer(base_model: str) -> Any:
    """Load the tokenizer and guarantee a pad token exists."""
    transformers = require("transformers", hint=torch_hint())
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_unsloth(
    *,
    base_model: str,
    max_seq_len: int,
    quant: str,
) -> tuple[Any, Any] | None:
    """Try to load through Unsloth; return ``None`` when unavailable/disabled."""
    import os

    if os.environ.get("FRICTIONTUNE_USE_UNSLOTH", "1") == "0":
        return None
    try:
        fast_model = require_attr(
            "unsloth", "FastLanguageModel", hint="pip install unsloth (optional)"
        )
    except Exception:
        return None
    model, tokenizer = fast_model.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=quant != "none",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_transformers_base(base_model: str, hardware: HardwareConfig) -> Any:
    """Load a causal LM with optional 4-bit quantisation (transformers path)."""
    torch = require("torch", hint=torch_hint())
    transformers = require("transformers", hint=torch_hint())
    quantization = None
    if hardware.quant in ("nf4", "fp4"):
        quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=hardware.quant,
            bnb_4bit_compute_dtype=compute_dtype(hardware, torch),
            bnb_4bit_use_double_quant=True,
        )
    return transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        torch_dtype=compute_dtype(hardware, torch),
        device_map="auto",
        trust_remote_code=True,
    )


def prepare_for_training(model: Any, hardware: HardwareConfig) -> Any:
    """Enable gradient checkpointing and input grads as the dtype requires.

    The kbit helper does both for a quantised model; a bf16/fp16 model must do
    them explicitly or the first backward pass fails on inputs that do not
    require grad. Kept in one place so every training entry point agrees.
    """
    peft = require("peft", hint=torch_hint())
    if hardware.quant in ("nf4", "fp4"):
        return peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=hardware.gradient_checkpointing
        )
    if hardware.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def apply_lora(
    model: Any,
    *,
    hardware: HardwareConfig,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    target_parameters: list[str],
) -> Any:
    """Prepare a quantised model and attach LoRA (MoE-safe by default)."""
    peft = require("peft", hint=torch_hint())
    model = prepare_for_training(model, hardware)
    config = peft.LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        target_parameters=target_parameters,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft.get_peft_model(model, config)


def continue_from_adapter(base_model: str, adapter_dir: str, hardware: HardwareConfig) -> Any:
    """Load a trained adapter for continued training (GRPO after SFT)."""
    peft = require("peft", hint=torch_hint())
    base = load_transformers_base(base_model, hardware)
    base = prepare_for_training(base, hardware)
    return peft.PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)


def save_adapter(model: Any, output_dir: str, tokenizer: Any) -> str:
    """Persist model+tokenizer and return the output directory."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return str(target)
