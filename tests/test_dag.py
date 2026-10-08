"""Tests for the pipeline dependency graph ordering.

The graph encodes ordering constraints that are not visible in any single stage:
if an edge is missing, a stage's run id silently ignores an artifact it consumes
or the stage is scheduled before its producer exists. These lock the edges that
matter for a real run.
"""

from __future__ import annotations

import pytest
from frictiontune.core.errors import ConfigError
from frictiontune.pipeline.dag import core_graph, full_graph


def test_train_grpo_closure_includes_its_adapter_producer_before_it() -> None:
    order = full_graph().closure(["train_grpo"])

    assert "train_sft" in order
    assert order.index("train_sft") < order.index("train_grpo")


def test_train_grpo_closure_pulls_in_judged_and_adapter_chain() -> None:
    closure = set(full_graph().closure(["train_grpo"]))

    # judged comes from judge; adapter from train_sft, which needs decontaminate.
    assert {"judge", "train_sft", "decontaminate"} <= closure


def test_core_graph_is_acyclic_and_orders_the_data_path() -> None:
    order = core_graph().order(
        [
            "ingest",
            "normalize",
            "segment",
            "friction_detect",
            "mine",
            "judge",
            "build_sft",
            "curate",
            "decontaminate",
            "train_sft",
            "evaluate_replay",
        ]
    )

    assert order.index("ingest") < order.index("evaluate_replay")
    assert order.index("judge") < order.index("build_sft") < order.index("curate")


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ConfigError):
        full_graph().order(["does_not_exist"])
