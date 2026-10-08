"""Dependency graph of the pipeline.

Edges are producer → consumer, derived from the artifacts each stage declares as
inputs. Where a stage also needs an ordering constraint with no shared artifact
(the verification and export tails), an explicit sequence edge is added. A stage
with no path from ``ingest`` cannot run; every core stage can reach
``evaluate_replay``.
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
    "friction_detect": ("attribute", "friction_cluster", "raft_srft"),
    "friction_cluster": ("synthesize",),
    "mine": ("synthesize",),
    "judge": ("train_grpo",),
    "train_sft": ("train_grpo", "rollout", "evaluate_sim_user", "evaluate_external"),
}

# Soft ordering edges: producer → consumer. They sequence the two stages when
# both are selected but do *not* create a dependency, so a core run of ``judge``
# is never forced to pull a Tier C producer in. They pair with a stage's
# ``optional_inputs``: `judge` folds in ``sft_raft``/``synthetic`` when the
# gated extensions that emit them have run.
OPTIONAL_EDGES: Mapping[str, tuple[str, ...]] = {
    "raft_srft": ("judge",),
    "synthesize": ("judge",),
}


def _merge_edges(
    base: Mapping[str, tuple[str, ...]], extra: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {node: tuple(targets) for node, targets in base.items()}
    for node, targets in extra.items():
        merged[node] = tuple(sorted(set(merged.get(node, ())) | set(targets)))
    return merged


class DependencyGraph:
    """A directed acyclic graph over stage names.

    Hard edges are producer → consumer dependencies: they are traversed by
    :meth:`closure`. Optional edges only constrain :meth:`order` (so a soft
    producer still runs before its consumer when both are selected); they are
    invisible to :meth:`closure` and :meth:`predecessors`.
    """

    def __init__(
        self,
        edges: Mapping[str, Iterable[str]],
        optional_edges: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        """Build the graph from producer-to-consumers adjacency maps."""
        self._edges: dict[str, tuple[str, ...]] = {
            node: tuple(sorted(set(targets))) for node, targets in edges.items()
        }
        ordering = dict(self._edges)
        for node, targets in (optional_edges or {}).items():
            ordering[node] = tuple(sorted(set(ordering.get(node, ())) | set(targets)))
        self._ordering = ordering
        self._nodes: set[str] = set(ordering) | {
            target for targets in ordering.values() for target in targets
        }

    def predecessors(self, node: str) -> tuple[str, ...]:
        """Direct upstream stages of ``node`` (hard dependencies only)."""
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
            for target in self._ordering.get(node, ()):
                if target in chosen:
                    indegree[target] += 1

        ready = sorted(name for name, degree in indegree.items() if degree == 0)
        ordered: list[str] = []
        while ready:
            node = ready.pop(0)
            ordered.append(node)
            for target in self._ordering.get(node, ()):
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
    return DependencyGraph(_merge_edges(CORE_EDGES, EXTENSION_EDGES), optional_edges=OPTIONAL_EDGES)
