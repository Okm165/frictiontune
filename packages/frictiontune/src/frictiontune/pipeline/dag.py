"""Dependency graph of the pipeline.

Edges are declared explicitly and validated against the registry so the graph
can never drift from the stages that actually exist. A stage with no path from
``ingest`` cannot run; every core stage can reach ``evaluate_replay``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frictiontune.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# Producer → consumer. The core data path plus the two prerequisite edges.
CORE_EDGES: Mapping[str, tuple[str, ...]] = {
    "fetch_base": ("train_sft",),
    "ingest": ("normalize",),
    "normalize": ("segment",),
    "segment": ("friction_detect",),
    "friction_detect": ("mine", "compile_rules"),
    "mine": ("judge",),
    "compile_rules": ("verify_rules",),
    "verify_rules": ("evaluate_replay",),
    "judge": ("build_sft",),
    "build_sft": ("curate",),
    "curate": ("decontaminate",),
    "decontaminate": ("train_sft",),
    "train_sft": ("evaluate_replay",),
    "evaluate_replay": ("export",),
}

# Extra edges that connect core stages to the Tier C extensions. These are kept
# separate so the core graph stays exactly the 15-stage data path described in
# the plan; the CLI orders *all* selected stages with :func:`full_graph`.
EXTENSION_EDGES: Mapping[str, tuple[str, ...]] = {
    "segment": ("reduce", "raft_srft"),
    "friction_detect": ("attribute", "friction_cluster"),
    "friction_cluster": ("synthesize",),
    "judge": ("synthesize", "train_grpo"),
    "curate": ("synthesize",),
    "train_sft": ("rollout", "evaluate_sim_user", "evaluate_external"),
}


def _merge_edges(
    base: Mapping[str, tuple[str, ...]], extra: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {node: tuple(targets) for node, targets in base.items()}
    for node, targets in extra.items():
        merged[node] = tuple(sorted(set(merged.get(node, ())) | set(targets)))
    return merged


class DependencyGraph:
    """A directed acyclic graph over stage names."""

    def __init__(self, edges: Mapping[str, Iterable[str]]) -> None:
        """Build the graph from a producer-to-consumers adjacency map."""
        self._edges: dict[str, tuple[str, ...]] = {
            node: tuple(sorted(set(targets))) for node, targets in edges.items()
        }
        self._nodes: set[str] = set(self._edges) | {
            target for targets in self._edges.values() for target in targets
        }

    def edges(self) -> Mapping[str, tuple[str, ...]]:
        """The adjacency map."""
        return self._edges

    def predecessors(self, node: str) -> tuple[str, ...]:
        """Direct upstream stages of ``node``."""
        return tuple(sorted(src for src, targets in self._edges.items() if node in targets))

    def order(self, selected: Sequence[str]) -> list[str]:
        """Topologically order ``selected``, respecting edges among them.

        Edges to stages outside ``selected`` are ignored, so a partial run (e.g.
        a single stage) is still well-defined. A cycle raises ``ConfigError``.
        """
        chosen = set(selected)
        unknown = chosen - self._nodes
        if unknown:
            raise ConfigError(f"unknown stage(s) in graph: {sorted(unknown)}")

        indegree: dict[str, int] = dict.fromkeys(chosen, 0)
        for node in chosen:
            for target in self._edges.get(node, ()):
                if target in chosen:
                    indegree[target] += 1

        ready = sorted(name for name, degree in indegree.items() if degree == 0)
        ordered: list[str] = []
        while ready:
            node = ready.pop(0)
            ordered.append(node)
            for target in self._edges.get(node, ()):
                if target in chosen:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        ready.append(target)
            ready.sort()

        if len(ordered) != len(chosen):
            cycle = sorted(chosen - set(ordered))
            raise ConfigError(f"dependency cycle among stages: {cycle}")
        return ordered

    def closure(self, selected: Sequence[str]) -> list[str]:
        """Expand ``selected`` to include all upstream dependencies, then order."""
        expanded = set(selected)
        frontier = list(selected)
        while frontier:
            node = frontier.pop()
            for upstream in self.predecessors(node):
                if upstream not in expanded:
                    expanded.add(upstream)
                    frontier.append(upstream)
        return self.order(sorted(expanded))


def core_graph() -> DependencyGraph:
    """The canonical core dependency graph."""
    return DependencyGraph(CORE_EDGES)


def full_graph() -> DependencyGraph:
    """The graph over every registered stage, core and extension alike."""
    return DependencyGraph(_merge_edges(CORE_EDGES, EXTENSION_EDGES))
