"""The base-model probe: the trainability guard for the training stages.

Quantized GGUF weights cannot be QLoRA-trained, and fused MoE experts silently
skip LoRA when ``target_parameters`` is misconfigured. The probe records what was
found so ``train_sft`` can refuse an impossible job instead of failing hours in.
"""

from __future__ import annotations

from pydantic import Field

from frictiontune.domain.trace import DomainModel


class BaseProbe(DomainModel):
    """What ``fetch_base`` learned about the configured base model."""

    repo_id: str
    path: str
    architecture: str = ""
    model_type: str = ""
    hidden_size: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    context_length: int = 0
    is_moe: bool = False
    num_experts: int = 0
    quantization: str = ""
    trainable: bool = False
    target_modules: list[str] = Field(default_factory=list)
    target_parameters: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
