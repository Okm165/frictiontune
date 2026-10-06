"""Guard: every structured call site supplies exactly its prompt's variables.

``render`` raises at call time when a prompt needs a variable the call site did
not pass, but only on the code path that actually runs. This test resolves every
``complete_structured`` call statically and compares the ``values`` it passes
with the placeholders in the named prompt, so a prompt and its call site cannot
drift apart unnoticed (the class of bug that once made ``evaluate_replay``
render ``judge`` without its threshold).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import frictiontune
from frictiontune.config.loader import load_config
from frictiontune.llm.prompts.registry import default_registry
from frictiontune.stages.common import analyst_prompt_values

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Map module-level ``NAME = "literal"`` assignments to their values."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if isinstance(value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value
    return constants


def _function_of(tree: ast.Module, lineno: int) -> ast.AST | None:
    """Return the innermost function containing ``lineno``."""
    found: ast.AST | None = None
    best = -1
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and node.lineno > best:
                found, best = node, node.lineno
    return found


def _local_dicts(function: ast.AST | None) -> dict[str, set[str]]:
    """Map ``NAME = {...}`` dict-literal assignments inside ``function`` to keys."""
    dicts: dict[str, set[str]] = {}
    if function is None:
        return dicts
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            dicts[node.targets[0].id] = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return dicts


def _resolve_str(arg: ast.expr, constants: dict[str, str]) -> str | None:
    """Resolve a string argument that is a literal or a module constant."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        return constants.get(arg.id)
    return None


def _resolve_keys(arg: ast.expr, dicts: dict[str, set[str]]) -> set[str] | None:
    """Resolve a ``values`` argument that is a dict literal or a named local."""
    if isinstance(arg, ast.Dict):
        return {
            key.value
            for key in arg.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    if isinstance(arg, ast.Name):
        return dicts.get(arg.id)
    return None


def test_call_sites_supply_exactly_their_prompt_variables() -> None:
    registry = default_registry()
    package_root = Path(frictiontune.__file__).parent
    checked = 0

    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _module_constants(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "complete_structured"
                and len(node.args) >= 4
            ):
                continue
            prompt = _resolve_str(node.args[2], constants)
            keys = _resolve_keys(node.args[3], _local_dicts(_function_of(tree, node.lineno)))
            if prompt is None or keys is None:
                continue
            template = registry.get(prompt)
            needed = set(_PLACEHOLDER.findall(template.system))
            needed |= set(_PLACEHOLDER.findall(template.user))
            where = f"{path.name}:{node.lineno} ({prompt})"
            assert not needed - keys, f"{where} is missing prompt variables {sorted(needed - keys)}"
            assert not keys - needed, f"{where} passes unused variables {sorted(keys - needed)}"
            checked += 1

    # Every structured call site that builds its values inline is checked here.
    # Call sites that delegate to a shared value-builder (the analyst) are
    # covered by test_analyst_values_match_the_analyst_prompt.
    assert checked >= 19


def test_analyst_values_match_the_analyst_prompt(config_dir: Path) -> None:
    template = default_registry().get("friction_analyst")
    needed = set(_PLACEHOLDER.findall(template.system))
    needed |= set(_PLACEHOLDER.findall(template.user))

    values = analyst_prompt_values(
        load_config(config_dir),
        session_id="s",
        episode_id="e",
        turn_count=1,
        tool_calls=0,
        tool_errors=0,
        compactions=0,
        transcript="0 | user | hi",
        max_chars=1000,
    )

    assert set(values) == needed


_WORKED_AXES = re.compile(
    r"form=(?P<form>[\w/-]+),\s*cause=(?P<cause>[\w/-]+),\s*"
    r"leaf=(?P<leaf>\w+),\s*cost=(?P<cost>[\w/-]+),\s*"
    r"resolution=(?P<resolution>[\w/-]+)"
)


def test_analyst_worked_example_axes_match_the_taxonomy(config_dir: Path) -> None:
    """The analyst's worked example must not contradict the injected taxonomy.

    The example freezes one concrete leaf→axes pair to teach the discipline. If
    the configured taxonomy moves that leaf's form/cause, the illustration would
    silently disagree with the ``leaf_axes`` block in the very same prompt, so
    the two are pinned together here.
    """
    taxonomy = load_config(config_dir).taxonomy
    text = default_registry().get("friction_analyst").system
    match = _WORKED_AXES.search(text)
    assert match is not None, "worked-example axes line not found; update this test"

    leaf = match["leaf"]
    assert leaf in taxonomy.leaves, f"example uses unknown leaf {leaf!r}"
    assert (match["form"], match["cause"]) == taxonomy.leaves[leaf]
    assert match["cost"] in taxonomy.costs
    assert match["resolution"] in taxonomy.resolutions
