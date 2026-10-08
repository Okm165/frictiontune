"""The stage contract.

Every stage declares its ROI tier, why it exists, its evidence, its cost, its
inputs/outputs, and the kill criterion that would retire it. A stage cannot be
registered without all of these, which is how the ROI discipline of the plan is
enforced in code rather than prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from frictiontune.domain.trace import DomainModel

Compute = Literal["cpu", "gpu", "llm", "mixed"]
Spend = Literal["none", "low", "medium", "high"]
HumanCost = Literal["none", "calibration-slice"]


class Tier(StrEnum):
    """ROI tier. Tier C is gated; Tier D is never run."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Lead(StrEnum):
    """Whether a stage is part of the default core or a demoted extension."""

    CORE = "core"
    EXTENSION = "extension"


class CostProfile(DomainModel):
    """Rough recurring cost of a stage."""

    compute: Compute = "cpu"
    api_usd: Spend = "none"
    human: HumanCost = "none"
    maintenance: Spend = "low"


class StageSpec(DomainModel):
    """The full, self-describing contract of a stage."""

    name: str
    tier: Tier
    lead: Lead
    roi: str
    justification: str
    evidence: list[str]
    cost: CostProfile
    inputs: dict[str, str] = Field(default_factory=dict)
    # Optional inputs never make a stage un-runnable: they are consumed (and
    # hashed into the run id) only when the artifact is present. They exist so a
    # core stage can fold in data produced by a gated extension without forcing
    # that extension on every run. Ordering is expressed with soft graph edges.
    optional_inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    kill_criterion: str = ""
    requires_llm: bool = False
    gated_by: str | None = None

    @property
    def core(self) -> bool:
        """Whether the stage belongs to the default core run."""
        return self.lead is Lead.CORE

    def input_label(self) -> str:
        """Human-readable input list; optional inputs are marked with ``?``."""
        parts = list(self.inputs)
        parts.extend(f"{name}?" for name in self.optional_inputs)
        return ", ".join(parts) or "-"
