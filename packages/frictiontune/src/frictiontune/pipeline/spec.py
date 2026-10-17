"""The stage contract.

Every stage declares its ROI tier, why it exists, its evidence, its cost, its
inputs/outputs, and the kill criterion that would retire it. A stage cannot be
registered without all of these, which is how the ROI discipline of the plan is
enforced in code rather than prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from frictiontune.config.models import RoleName
from frictiontune.domain.trace import DomainModel

Compute = Literal["cpu", "gpu", "llm", "mixed"]
Spend = Literal["none", "low", "medium", "high"]
HumanCost = Literal["none", "calibration-slice"]


class Tier(StrEnum):
    """ROI tier: A runs by default, C is gated, D is never run."""

    A = "A"
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
    # The roles this stage actually calls. A stage's run identity must depend on
    # the models *it* uses, not on every model the run happens to configure: a
    # phase that starts the base and adapter servers for the replay changes the
    # `target` and `adapter` bindings, and without this the whole upstream data
    # chain would re-key and rebuild. Leave empty for a stage that calls no
    # model (`requires_llm` false); a stage that does call models but names none
    # is fingerprinted against every role, which is safe but over-broad.
    roles: list[RoleName] = Field(default_factory=list)
    # The prompts this stage renders, by library name. A stage's run identity
    # must depend on the prompts *it* uses, not on the whole library: otherwise
    # editing `judge.md` re-keys `ingest`. Leave empty for a stage that calls no
    # model; a stage that does call models but names none is fingerprinted
    # against the whole library, which is safe but over-broad.
    prompts: list[str] = Field(default_factory=list)
    # Whether the stage consumes the ``evaluation`` policy block (the time-split
    # quantile, the contamination tiers, the replay seeds). It is in the
    # fingerprint only for the stages that read it: leaving it global meant
    # changing a *measurement* policy re-keyed the whole data phase, so the one
    # experiment that improves the instrument — holding out more sessions —
    # cost a full rebuild. Note that ``taxonomy``, ``hardware`` and ``embedding``
    # are still global in the same way; they change far less often, but the same
    # scoping would apply.
    uses_evaluation: bool = False
    gated_by: str | None = None

    @model_validator(mode="after")
    def _declare_report(self) -> StageSpec:
        """Declare the human-readable ``report`` every stage emits.

        The report is a universal output, so listing it in each spec drifted:
        it was emitted by every stage but declared by none (GAP-005). Declaring
        it once here keeps the contract truthful and stops the drift recurring.
        """
        self.outputs.setdefault("report", "text")
        return self

    @property
    def core(self) -> bool:
        """Whether the stage belongs to the default core run."""
        return self.lead is Lead.CORE

    def input_label(self) -> str:
        """Human-readable input list; optional inputs are marked with ``?``."""
        parts = list(self.inputs)
        parts.extend(f"{name}?" for name in self.optional_inputs)
        return ", ".join(parts) or "-"
