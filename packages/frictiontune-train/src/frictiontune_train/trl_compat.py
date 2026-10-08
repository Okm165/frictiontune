"""Resolve frictiontune's frozen trainer settings against the installed TRL.

TRL renames its public arguments between releases:

* ``SFTTrainer(tokenizer=...)`` / ``GRPOTrainer(tokenizer=...)`` became
  ``processing_class=...`` in 0.12 and the old spelling is removed in 2.0.
* ``SFTConfig.max_seq_length`` became ``max_length``.
* ``GRPOConfig`` has always used ``max_completion_length`` where a naive port
  from ``generate`` reaches for ``max_new_tokens``.

Pinning one release's spelling makes a real run fail — or, worse, silently
ignore a knob — the moment the environment resolves a different TRL. These
helpers resolve the frozen set of settings frictiontune needs to whatever the
installed TRL actually accepts, and raise a named stage error when a setting
cannot be expressed at all, so a version mismatch is diagnosed rather than
mis-attributed to the data or the model.

This is deterministic plumbing: it inspects signatures and maps names. It makes
no semantic decision and performs no retry.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from frictiontune.core.errors import StageError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _field_names(cls: type[Any]) -> frozenset[str]:
    """Return the keyword names ``cls`` accepts in its constructor.

    Dataclasses are read from ``__dataclass_fields__`` (so inherited
    ``TrainingArguments`` fields are included and ``init=False`` fields are
    excluded); anything else falls back to its ``__init__`` signature.
    """
    dataclass_fields: Any = getattr(cls, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, dict):
        return frozenset(
            name for name, field in dataclass_fields.items() if getattr(field, "init", True)
        )
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - exotic classes
        return frozenset()
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    )


def resolve_config_kwargs(
    config_cls: type[Any],
    settings: Mapping[str, Any],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
    stage: str,
) -> dict[str, Any]:
    """Map logical settings to the constructor kwargs ``config_cls`` accepts.

    Each setting key is resolved to the first name in ``aliases[key]`` (or the
    key itself) that exists on ``config_cls``. A setting that matches no name is
    a hard error: dropping it silently would let a configured knob — a sequence
    length, a generation budget — have no effect on training.
    """
    accepted = _field_names(config_cls)
    aliases = aliases or {}
    resolved: dict[str, Any] = {}
    for name, value in settings.items():
        candidates = aliases.get(name, (name,))
        for candidate in candidates:
            if candidate in accepted:
                resolved[candidate] = value
                break
        else:
            supported = ", ".join(repr(candidate) for candidate in candidates)
            raise StageError(
                stage,
                f"{config_cls.__name__} accepts none of {supported}; the installed "
                "TRL/transformers is incompatible with this stage",
            )
    return resolved


def trainer_token_kwargs(trainer_cls: type[Any], tokenizer: Any, *, stage: str) -> dict[str, Any]:
    """Return the trainer kwarg that supplies the tokenizer/processing class.

    TRL renamed ``tokenizer`` to ``processing_class`` in 0.12 and removes the
    old name in 2.0, so pick whichever the installed trainer declares.
    """
    accepted = _field_names(trainer_cls)
    if "processing_class" in accepted:
        return {"processing_class": tokenizer}
    if "tokenizer" in accepted:
        return {"tokenizer": tokenizer}
    raise StageError(
        stage,
        f"{trainer_cls.__name__} declares neither 'processing_class' nor 'tokenizer'; "
        "the installed TRL is incompatible with this stage",
    )
