"""Tests for the capability probe (the forgetting instrument, GAP-094).

Only the deterministic plumbing is unit-tested here; the generation and judging
need the base/adapter servers and are exercised by a replay-phase run.
"""

from __future__ import annotations

from frictiontune.domain.responses import ProbeVerdict
from frictiontune.stages._extensions import (
    _PROBE_PROMPTS,
    CapabilityProbe,
)


def test_probe_prompts_are_a_fixed_corpus_independent_instrument() -> None:
    # The prompts are the instrument: fixed, distinct, self-contained, and
    # deliberately not drawn from the corpus (so tuning on the corpus cannot
    # teach the probe's answers).
    assert len(_PROBE_PROMPTS) >= 6
    assert len(set(_PROBE_PROMPTS)) == len(_PROBE_PROMPTS)
    assert all(prompt.strip() for prompt in _PROBE_PROMPTS)
    assert all(len(prompt) < 1_000 for prompt in _PROBE_PROMPTS)


def test_probe_measures_regression_axes_without_friction_ceremony() -> None:
    # No friction_avoidance axis: these prompts have no friction context, and a
    # field the stage never reads is a place for the judge to run away.
    signal_names = {name for name, _dir in CapabilityProbe._SIGNALS}
    assert signal_names == {"correctness", "clarity", "verbosity_chars", "format_valid"}


def test_probe_verdict_carries_only_what_the_probe_reads() -> None:
    verdict = ProbeVerdict(reason="fine", correctness=3, clarity=4)
    assert verdict.correctness == 3
    assert verdict.clarity == 4
    assert set(ProbeVerdict.model_fields) == {"reason", "correctness", "clarity"}


def test_probe_spec_is_gated_and_needs_both_servers() -> None:
    spec = CapabilityProbe.spec
    assert spec.tier.value == "C"
    assert spec.requires_llm
    assert set(spec.roles or ()) == {"target", "adapter", "judge"}
    assert "adapter" in spec.inputs
    assert spec.gated_by is not None
