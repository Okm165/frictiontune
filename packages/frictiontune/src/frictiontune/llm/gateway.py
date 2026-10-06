"""The single funnel for every LLM call in frictiontune.

All model access — text or structured — passes through :class:`LlmGateway`.
It composes exactly three primitives, each defined once:

* :class:`~frictiontune.llm.client.LlmClient` — transport, endpoint caching,
  and *transport* retries (via the shared :func:`with_retry`).
* :func:`~frictiontune.llm.prompts.registry.PromptRegistry` — versioned prompts.
* :func:`with_retry` — the one retry mechanism, reused here for *schema* repair.

Structured calls are validated into strict pydantic models; a validation or
parse failure is surfaced as :class:`LlmSchemaError`, fed back to the model, and
retried under the same generic policy. No stage ever implements its own retry.

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
from frictiontune.llm.parsing import extract_json_object

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    "Schema below: use the exact field names, types, and enum values, include every "
    "required field, and add no field the schema does not define. Reply with the "
    "JSON object only — no prose and no markdown fences."
)


def schema_contract(schema: type[BaseModel]) -> str:
    """Render the authoritative output contract for ``schema``.

    The pydantic model is the source of truth, so the JSON Schema is generated
    from it rather than duplicated by hand in each prompt.
    """
    return f"{_SCHEMA_CONTRACT}\n\nJSON Schema:\n{dumps(schema.model_json_schema())}"


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
        self._repair_policy = RetryPolicy(
            max_attempts=max_repairs + 1,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
        )

    async def complete_text(
        self,
        role: RoleName,
        config: LlmRoleConfig,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LlmResponse:
        """Raw completion (still transport-retried and cached by the client)."""
        return await self._client.complete(role, config, messages, tools=tools)

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
        history: list[ChatMessage] = [
            ChatMessage.system(system),
            ChatMessage.system(schema_contract(schema)),
            ChatMessage.user(user),
        ]

        async def attempt() -> ModelT:
            response = await self._client.complete(
                role,
                config,
                history,
                force_json=True,
                response_schema=schema.model_json_schema(),
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
        """Return a model instance, or a human-readable problem string."""
        parsed = extract_json_object(text)
        if parsed is None:
            return "reply contained no JSON object"
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            return exc.json(include_url=False)
