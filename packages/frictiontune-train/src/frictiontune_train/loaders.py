"""Single definition of model/tokenizer loading for SFT and GRPO.

Both training paths must load the base model, apply the same quantisation, and
attach LoRA the same way; duplicating that logic is how MoE and dtype bugs creep
in. This module owns it once.
"""

from __future__ import annotations

import os
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


def load_transformers_base(
    base_model: str, hardware: HardwareConfig, *, device_map: Any = "auto"
) -> Any:
    """Load a causal LM with optional 4-bit quantisation (transformers path).

    ``device_map`` decides placement. ``"auto"`` is right for a single process
    that sees one card and for inference/merge with every card visible (where
    ``auto`` shards as intended). Under DDP each rank sees every card but must
    hold the *whole* model on its own, so :mod:`frictiontune_train.distributed`
    passes ``{"": local_rank}`` instead.
    """
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
        **_dtype_kwarg(transformers, compute_dtype(hardware, torch)),
        device_map=device_map,
        attn_implementation=_attention_implementation(transformers),
        trust_remote_code=True,
    )


def _attention_implementation(transformers: Any) -> str:
    """Choose the attention kernel, preferring a GQA-safe fused SDPA.

    Grouped-query attention (Qwen3: 32 query heads, 8 KV heads) reaches
    ``scaled_dot_product_attention`` as ``enable_gqa``. The ROCm build's fused
    kernels reject that broadcast and silently fall back to the **math** kernel,
    which materialises the full ``[batch, heads, seq, seq]`` score matrix — more
    than ten gigabytes at an 8k window, enough to OOM a 24 GB card on the first
    step. Transformers takes the ``enable_gqa`` path only when it sees *no*
    attention mask, and it optimises away an all-ones mask, so a full-length
    batch (no padding) always trips it.

    The registered ``sdpa_gqa`` variant repeats the KV heads up to the query head
    count — a few megabytes — before delegating to the stock SDPA function, so
    the fused kernel applies instead of the math one. Registration goes through
    the public ``AttentionInterface``, so a model asks for it by name exactly
    like a built-in. ``FRICTIONTUNE_ATTN_IMPL`` overrides the choice (for
    example ``eager`` when debugging); a transformers that lacks the interface
    falls back to plain ``sdpa``.
    """
    override = os.environ.get("FRICTIONTUNE_ATTN_IMPL")
    if override:
        return override
    try:
        interface = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
        if "sdpa_gqa" not in interface.valid_keys():
            from transformers.integrations.sdpa_attention import (
                repeat_kv,
                sdpa_attention_forward,
            )

            def sdpa_gqa(
                module: Any,
                query: Any,
                key: Any,
                value: Any,
                attention_mask: Any = None,
                dropout: float = 0.0,
                scaling: Any = None,
                is_causal: Any = None,
                **kwargs: Any,
            ) -> Any:
                groups = getattr(module, "num_key_value_groups", 1)
                if groups > 1 and key.shape[1] != query.shape[1]:
                    key = repeat_kv(key, groups)
                    value = repeat_kv(value, groups)
                    keep = module.num_key_value_groups
                    module.num_key_value_groups = 1
                    try:
                        return sdpa_attention_forward(
                            module,
                            query,
                            key,
                            value,
                            attention_mask,
                            dropout,
                            scaling,
                            is_causal,
                            **kwargs,
                        )
                    finally:
                        module.num_key_value_groups = keep
                return sdpa_attention_forward(
                    module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
                )

            interface.register("sdpa_gqa", sdpa_gqa)
        return "sdpa_gqa"
    except Exception:  # the fused kernel is an optimisation, not a requirement
        return "sdpa"


def _dtype_kwarg(transformers: Any, dtype: Any) -> dict[str, Any]:
    """Spell the model dtype kwarg for the installed transformers.

    transformers 5 renamed ``torch_dtype`` to ``dtype`` and warns the old name
    will be removed; use whichever the installed major version documents.
    """
    major = int(str(transformers.__version__).split(".", 1)[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


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


def continue_from_adapter(
    base_model: str, adapter_dir: str, hardware: HardwareConfig, *, device_map: Any = "auto"
) -> Any:
    """Load a trained adapter for continued training (GRPO after SFT)."""
    peft = require("peft", hint=torch_hint())
    base = load_transformers_base(base_model, hardware, device_map=device_map)
    base = prepare_for_training(base, hardware)
    return peft.PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)


def save_adapter(model: Any, output_dir: str, tokenizer: Any) -> str:
    """Persist model+tokenizer and return the output directory."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return str(target)
