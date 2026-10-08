"""The single funnel for every LLM call in frictiontune.

All model access — text or structured — passes through :class:`LlmGateway`.
It composes exactly three primitives, each defined once:

* :class:`~frictiontune.llm.client.LlmClient` — transport, endpoint caching,
  and *transport* retries (via the shared :func:`with_retry`).
* :func:`~frictiontune.llm.prompts.registry.PromptRegistry` — versioned prompts.
* :func:`with_retry` — the one retry mechanism, reused here for *schema* repair.

Structured calls are validated into strict pydantic models; a validation or
parse failure is surfaced as :class:`LlmSchemaError`, fed back to the model, and
retried under the same generic policy. No stage ever implements its own retry,
so every LLM call retries through exactly one mechanism.

The output contract is *derived from the schema*, never hand-written in the
prompt: :func:`schema_contract` renders the model's JSON Schema and it is sent as
an extra system message. The pydantic model is therefore the single source of
truth for field names, types, and enums, so a prompt cannot drift from the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from frictiontune.core.errors import LlmSchemaError
from frictiontune.core.errors import ValidationError as FtValidationError
from frictiontune.core.json import dumps
from frictiontune.core.logging import get_logger
from frictiontune.core.retry import RetryPolicy, with_retry
from frictiontune.llm.messages import ChatMessage, ToolSpec
from frictiontune.llm.parsing import iter_json_objects

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from frictiontune.config.models import LlmRoleConfig, RoleName
    from frictiontune.llm.client import LlmClient, LlmResponse
    from frictiontune.llm.prompts.registry import PromptRegistry

ModelT = TypeVar("ModelT", bound=BaseModel)

_log = get_logger(__name__)

_REPAIR_PREFIX = (
    "The previous reply did not satisfy the required JSON schema. "
    "Reply with ONLY one JSON object that conforms exactly. No prose, no code fences."
)

_SCHEMA_CONTRACT = (
    "Your reply MUST be a single JSON object that conforms exactly to the JSON "
    "Schema below: use the exact field names, types, and enum values, write "
    "numbers and booleans as JSON literals (5, 0.8, true — never as strings), "
    "include every required field, and add no field the schema does not define. "
    "Reply with the JSON object only — no prose and no markdown fences."
)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a self-contained copy of ``schema`` for the model to read.

    Pydantic renders nested response models as ``$defs``/``$ref``, but local
    models often ignore ``$defs``. Inline every reference that is not part of a
    recursive cycle; only genuinely recursive definitions keep their ``$defs``.
    ``title`` keys are dropped as noise. Key order is preserved, so the
    deliberate reason-before-decision field order survives.
    """
    defs = schema.get("$defs")
    definitions = defs if isinstance(defs, dict) else {}
    kept_ref = [False]
    inlined = {
        key: _resolve(value, definitions, frozenset(), kept_ref)
        for key, value in schema.items()
        if key not in ("$defs", "title")
    }
    if kept_ref[0]:
        inlined["$defs"] = definitions
    return inlined


def _resolve(node: Any, defs: dict[str, Any], stack: frozenset[str], kept_ref: list[bool]) -> Any:
    """Recursively inline ``$ref`` nodes, protecting against reference cycles."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            target = defs.get(name)
            if name in stack or not isinstance(target, dict):
                kept_ref[0] = True
                return dict(node)
            return _resolve(target, defs, stack | {name}, kept_ref)
        return {
            key: _resolve(value, defs, stack, kept_ref)
            for key, value in node.items()
            if key != "title"
        }
    if isinstance(node, list):
        return [_resolve(item, defs, stack, kept_ref) for item in node]
    return node


def schema_contract(schema: type[BaseModel]) -> str:
    """Render the authoritative output contract for ``schema``.

    The pydantic model is the source of truth, so the JSON Schema is generated
    from it rather than duplicated by hand in each prompt. It is inlined and
    kept in declaration order so the model sees a self-contained contract whose
    field order matches the reasoning order the prompt asks for.
    """
    compact = _inline_refs(schema.model_json_schema())
    body = dumps(compact, sort_keys=False)
    return f"{_SCHEMA_CONTRACT}\n\nJSON Schema:\n{body}"


class LlmGateway:
    """One object through which all LLM traffic flows."""

    def __init__(
        self,
        client: LlmClient,
        registry: PromptRegistry,
        *,
        max_repairs: int = 3,
    ) -> None:
        """Store the client and prompt registry, and the repair budget."""
        self._client = client
        self._registry = registry
        self._max_repairs = max_repairs
        self._repair_policy = RetryPolicy.immediate(max_repairs + 1)

    def prompt_fingerprint(self) -> str:
        """Fingerprint of the on-disk prompt library, for run identity."""
        return self._registry.fingerprint()

    async def complete_text(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        seed: int | None = None,
    ) -> LlmResponse:
        """Raw completion (still transport-retried and cached by the client)."""
        return await self._client.complete(role, config, messages, tools=tools, seed=seed)

    async def complete_structured(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        prompt_name: str,
        values: dict[str, object],
        schema: type[ModelT],
    ) -> ModelT:
        """Render ``prompt_name`` and return a validated ``schema`` instance."""
        template = self._registry.get(prompt_name)
        if template.schema_name and template.schema_name != schema.__name__:
            _log.warning(
                "prompt declares a different schema than the call site",
                prompt=prompt_name,
                declared=template.schema_name,
                actual=schema.__name__,
            )
        system, user = template.render(values)
        # One system message, not two: several OpenAI-compatible servers reject
        # (or silently mishandle) multiple system turns. The schema contract is
        # folded into the prompt's system text so every endpoint sees one.
        history: list[ChatMessage] = [
            ChatMessage.system(f"{system}\n\n{schema_contract(schema)}"),
            ChatMessage.user(user),
        ]

        async def attempt() -> ModelT:
            response = await self._client.complete(
                role,
                config,
                history,
                force_json=True,
                response_schema=_inline_refs(schema.model_json_schema()),
                schema_name=schema.__name__,
            )
            problem = self._validate(response.text, schema)
            if not isinstance(problem, str):
                return problem
            _log.warning(
                "structured output rejected",
                role=role.value,
                prompt=prompt_name,
                schema=schema.__name__,
            )
            history.append(ChatMessage.assistant(response.text))
            history.append(ChatMessage.user(f"{_REPAIR_PREFIX}\n\nProblem:\n{problem}"))
            raise LlmSchemaError(f"{role.value}/{prompt_name}: {problem}")

        try:
            return await with_retry(
                attempt,
                policy=self._repair_policy,
                retry_on=(LlmSchemaError,),
                logger=_log,
            )
        except LlmSchemaError as exc:
            raise FtValidationError(schema.__name__, self._max_repairs + 1, str(exc)) from exc

    @staticmethod
    def _validate(text: str, schema: type[ModelT]) -> ModelT | str:
        """Return a model instance, or a human-readable problem string.

        Validation is strict: a stringified number or boolean is a malformed
        reply, not something to coerce. A reply may contain more than one
        top-level object; the first candidate that validates against the schema
        wins, because the contract asks the model for exactly one object. The
        problem string is concise and fed back to the model by the repair loop.
        """
        candidates = list(iter_json_objects(text))
        if not candidates:
            return "reply contained no JSON object"
        last_error: ValidationError | None = None
        for parsed in candidates:
            try:
                return schema.model_validate(parsed, strict=True)
            except ValidationError as exc:
                last_error = exc
        assert last_error is not None  # candidates is non-empty and all failed
        return _concise_errors(last_error)


def _concise_errors(exc: ValidationError) -> str:
    """Render a pydantic error as ``path: message`` lines, without echoing input."""
    lines: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"{location or '<root>'}: {error.get('msg', 'invalid')}")
    return "; ".join(lines) or "reply did not match the schema"
