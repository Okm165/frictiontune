"""Reward composition for GRPO.

The reward is deliberately *plural*: a single scalar proxy is easy to hack, so
GRPO here optimises a weighted sum of independent components, each of which can
be inspected in the run report.

Components:

* ``judge`` — a required remote LLM/judge score. A judge endpoint must be
  configured: without one the remaining terms are constant for well-formed text
  and GRPO would optimise nothing. Refusing to run is better than a silent no-op.
* ``format`` — deterministic plumbing: the completion is well-formed and not
  degenerate (empty, runaway length).
* ``tool`` — deterministic plumbing: the fenced JSON tool calls parse. A
  completion that makes no tool call scores zero on this term; it measures the
  tool-call protocol, not meaning.

Only the judge component makes a semantic call; the rest are parser/validator
checks, consistent with the project's "LLM judges, deterministic plumbing" rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from frictiontune.core.errors import ConfigError
from frictiontune.core.json import json_block_validity, json_blocks

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_OPEN_MARKER = "<|im_start|>"
_CLOSE_MARKER = "<|im_end|>"


def _format_score(completion: str, *, max_chars: int) -> float:
    """1.0 for a well-formed completion, 0.0 for a degenerate one."""
    text = completion.strip()
    if not text:
        return 0.0
    if len(text) > max_chars:
        return 0.0
    if text.count(_OPEN_MARKER) and text.count(_CLOSE_MARKER) < text.count(_OPEN_MARKER):
        return 0.0
    return 1.0


def _tool_score(completion: str) -> float:
    """Validity of the tool calls the completion chose to emit.

    A completion that makes no fenced JSON tool call scores zero: this term
    rewards a well-formed tool-call protocol, and a plain text answer has none.
    When a JSON fence is present, the score is the fraction of those fenced JSON
    blocks that parse. Non-JSON fences (```python, …) are code, not a tool call,
    and are ignored.
    """
    if not json_blocks(completion):
        return 0.0
    return json_block_validity(completion)


class RewardComposer:
    """A callable GRPO reward combining a required judge with format/tool terms."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        judge_url: str | None = None,
        judge_timeout_s: float = 30.0,
        max_chars: int = 40_000,
    ) -> None:
        """Initialise the composer; a judge endpoint is required."""
        if not judge_url:
            raise ConfigError(
                "GRPO reward requires a judge endpoint; set "
                "FRICTIONTUNE_GRPO_JUDGE_URL so the reward makes a semantic "
                "judgement instead of a constant that optimises nothing"
            )
        self._weights = weights or {"judge": 1.0, "format": 0.5, "tool": 0.5}
        self._judge_url = judge_url
        self._judge_timeout = judge_timeout_s
        self._max_chars = max_chars

    def _judge(self, prompt: str, completion: str) -> float:
        """Query the judge endpoint; an unparseable reply scores 0.0."""
        import httpx
        from frictiontune.core.retry import RetryPolicy, with_retry
        from frictiontune.llm.transport import is_transport_retryable

        url = self._judge_url

        def _post() -> httpx.Response:
            response = httpx.post(
                url,
                json={"prompt": prompt, "completion": completion},
                timeout=self._judge_timeout,
            )
            response.raise_for_status()
            return response

        response = with_retry(_post, policy=RetryPolicy(), retry_on=is_transport_retryable)
        try:
            payload = response.json()
        except ValueError:
            # A judge that answers with prose or an error page must not abort
            # the whole training run; an unparseable reply scores zero.
            return 0.0
        if isinstance(payload, dict) and isinstance(payload.get("score"), (int, float)):
            return float(payload["score"])
        return 0.0

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        """Score a group of completions (TRL ``reward_funcs`` signature)."""
        prompts: Sequence[str] = kwargs.get("prompts") or [""] * len(completions)
        scores: list[float] = []
        for completion, prompt in zip(completions, prompts, strict=False):
            total = 0.0
            total += self._weights.get("format", 0.0) * _format_score(
                completion, max_chars=self._max_chars
            )
            total += self._weights.get("tool", 0.0) * _tool_score(completion)
            judge_weight = self._weights.get("judge", 0.0)
            if judge_weight:
                total += judge_weight * self._judge(prompt, completion)
            scores.append(total)
        return scores


def build_reward(
    weights: dict[str, float], *, judge_url: str | None = None
) -> Callable[..., list[float]]:
    """Return a GRPO reward function for the given weights."""
    return RewardComposer(weights=weights, judge_url=judge_url)
