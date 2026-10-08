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


def test_judge_hard_dependencies_are_mined_candidates_only() -> None:
    # The gated producers are *optional* inputs, so they must not become hard
    # dependencies that a core closure is forced to pull in.
    assert full_graph().predecessors("judge") == ("mine",)


def test_optional_edges_do_not_force_gated_producers_into_a_closure() -> None:
    closure = set(full_graph().closure(["judge"]))

    assert "judge" in closure
    assert {"synthesize", "raft_srft", "friction_cluster"}.isdisjoint(closure)


def test_optional_edges_still_order_producers_before_the_judge() -> None:
    order = full_graph().order(
        [
            "ingest",
            "normalize",
            "segment",
            "friction_detect",
            "mine",
            "friction_cluster",
            "synthesize",
            "raft_srft",
            "judge",
            "build_sft",
        ]
    )

    assert order.index("raft_srft") < order.index("judge")
    assert order.index("synthesize") < order.index("judge")
    assert order.index("friction_cluster") < order.index("synthesize")


def test_core_graph_has_no_extension_ordering() -> None:
    assert "synthesize" not in core_graph().predecessors("judge")
    assert "raft_srft" not in core_graph().predecessors("judge")
