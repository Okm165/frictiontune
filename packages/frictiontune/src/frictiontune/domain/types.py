"""Shared field types for LLM/agent responses.

Agent responses are validated in *strict* mode (see ``LlmGateway``), so pydantic
rejects the coercions it allows by default: a string is not an integer, a string
is not a boolean, and so on. That is the point — a malformed reply is surfaced
and repaired instead of being silently massaged into shape.

Two JSON realities need explicit allowances, and only these:

* JSON has a single number type, so an integer literal such as ``1`` is a valid
  ``0-1`` probability. ``Probability`` keeps the range constraint while accepting
  integral numbers.
* JSON carries an enum's *value*, not its member. ``enum_by_value`` maps a
  matching value to the member so enum fields stay strict for everything else.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, TypeVar

from pydantic import BeforeValidator, Field

EnumT = TypeVar("EnumT", bound=Enum)


def _to_int(value: Any) -> Any:
    """Coerce an integral JSON number to ``int``; leave everything else alone.

    JSON has a single number type, so a model may write a count, index, or 0-5
    score as ``4.0`` rather than ``4``. Both are the same JSON number. A
    non-integral float, a string, or a bool is left untouched so strict
    validation still rejects it (a stringified number is a malformed reply).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# JSON numbers are not typed, so an integer field accepts an integral float.
# The counterpart of ``Probability`` for count/index/score fields.
IntLike = Annotated[int, BeforeValidator(_to_int)]

# A count or turn index: integral-float tolerant and non-negative.
NonNegativeInt = Annotated[int, BeforeValidator(_to_int), Field(ge=0)]

# A 0-5 rubric score: integral-float tolerant, still bounded.
Score = Annotated[int, BeforeValidator(_to_int), Field(ge=0, le=5)]

Probability = Annotated[float, Field(ge=0.0, le=1.0, strict=False)]


def enum_by_value(enum_type: type[EnumT]) -> BeforeValidator:
    """Build a validator that accepts ``enum_type`` by value, not just member.

    Strict validation demands an actual enum member, but a JSON reply only ever
    contains the value. This coerces a matching value into its member before the
    strict check; an unknown value is left untouched and fails validation as it
    should.
    """

    def _coerce(value: object) -> object:
        if isinstance(value, str) and not isinstance(value, enum_type):
            try:
                return enum_type(value)
            except ValueError:
                return value
        return value

    return BeforeValidator(_coerce)
