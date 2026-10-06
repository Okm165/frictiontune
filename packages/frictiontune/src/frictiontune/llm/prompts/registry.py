"""Versioned prompt registry.

Prompts are data, not inline strings. Each prompt lives in
``prompts/library/<name>.md`` as a small document:

* a YAML front matter block (``name``, ``version``, ``role``, ``schema``,
  ``description``),
* one or more fenced sections introduced by ``:::system`` / ``:::user`` / ``:::notes``.

Placeholders are written ``{{ variable }}`` and rendered strictly; a missing
placeholder is an error, never an empty string, so prompts cannot silently
degrade.

Boilerplate shared by many prompts lives once under ``library/_partials/`` and is
pulled in with ``{{> partial-name }}``. Includes are expanded at parse time, so
a prompt always renders to concrete text and a missing or cyclic partial is a
hard error rather than a silent no-op.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from frictiontune.config.models import RoleName
from frictiontune.core.errors import PromptError

if TYPE_CHECKING:
    from collections.abc import Mapping

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_SECTION = re.compile(r"^:::(\w+)\s*\n(.*?)^:::\s*$", re.DOTALL | re.MULTILINE)
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_INCLUDE = re.compile(r"\{\{>\s*([A-Za-z0-9_./-]+)\s*\}\}")


def render(template: str, values: dict[str, Any]) -> str:
    """Substitute ``{{ key }}`` placeholders, failing on any missing key."""

    def _substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise PromptError(f"missing prompt variable: {key!r}")
        return str(values[key])

    return _PLACEHOLDER.sub(_substitute, template)


def _expand_includes(
    text: str, partials: Mapping[str, str], *, stack: frozenset[str] = frozenset()
) -> str:
    """Replace ``{{> name }}`` includes with the named partial, recursively."""

    def _substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in stack:
            raise PromptError(f"cyclic prompt include: {name!r}")
        if name not in partials:
            raise PromptError(f"unknown prompt partial: {name!r}")
        return _expand_includes(partials[name], partials, stack=stack | {name})

    return _INCLUDE.sub(_substitute, text)


class PromptTemplate(BaseModel):
    """A parsed, renderable prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    role: RoleName | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    description: str = ""
    sections: dict[str, str] = Field(default_factory=dict)

    @property
    def system(self) -> str:
        """The system-prompt section (empty if the prompt has none)."""
        return self.sections.get("system", "")

    @property
    def user(self) -> str:
        """The user-prompt section (empty if the prompt has none)."""
        return self.sections.get("user", "")

    def render(self, values: dict[str, Any]) -> tuple[str, str]:
        """Render ``(system, user)`` for the supplied variable values."""
        return render(self.system, values), render(self.user, values)

    @classmethod
    def parse(cls, text: str, *, partials: Mapping[str, str] | None = None) -> PromptTemplate:
        """Parse a prompt document into a template, expanding ``{{> }}`` includes."""
        match = _FRONT_MATTER.match(text)
        if not match:
            raise PromptError("prompt is missing a YAML front matter block")
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise PromptError(f"invalid prompt front matter: {exc}") from exc
        body = match.group(2)
        resolved = partials or {}
        sections = {
            key: _expand_includes(value.strip(), resolved) for key, value in _SECTION.findall(body)
        }
        if "system" not in sections and "user" not in sections:
            raise PromptError("prompt defines neither a :::system nor a :::user section")
        try:
            return cls.model_validate({**meta, "sections": sections})
        except Exception as exc:
            raise PromptError(f"invalid prompt metadata: {exc}") from exc


class PromptRegistry:
    """Loads and caches prompt templates from the packaged library."""

    def __init__(self, package: str = "frictiontune.llm.prompts") -> None:
        """Load prompt documents from ``package/library``."""
        self._dir = files(package).joinpath("library")
        self._cache: dict[str, PromptTemplate] = {}
        self._partials_cache: dict[str, str] | None = None

    def names(self) -> list[str]:
        """List the names of every available prompt."""
        return sorted(
            path.name.removesuffix(".md")
            for path in self._dir.iterdir()
            if path.name.endswith(".md")
        )

    def partials(self) -> dict[str, str]:
        """Load the shared ``library/_partials/*.md`` fragments, once."""
        if self._partials_cache is None:
            loaded: dict[str, str] = {}
            partial_dir = self._dir.joinpath("_partials")
            if partial_dir.is_dir():
                for path in partial_dir.iterdir():
                    if path.name.endswith(".md"):
                        loaded[path.name.removesuffix(".md")] = path.read_text(
                            encoding="utf-8"
                        ).strip()
            self._partials_cache = loaded
        return self._partials_cache

    def get(self, name: str) -> PromptTemplate:
        """Return a parsed prompt template by name."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        resource = self._dir.joinpath(f"{name}.md")
        if not resource.is_file():
            raise PromptError(f"unknown prompt: {name!r}")
        template = PromptTemplate.parse(
            resource.read_text(encoding="utf-8"), partials=self.partials()
        )
        self._cache[name] = template
        return template


@lru_cache(maxsize=1)
def default_registry() -> PromptRegistry:
    """Return the process-wide prompt registry."""
    return PromptRegistry()
