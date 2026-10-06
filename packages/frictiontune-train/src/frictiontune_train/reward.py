"""Reward composition for GRPO.

The reward is deliberately *plural*: a single scalar proxy is easy to hack, so
GRPO here optimises a weighted sum of independent components, each of which can
be inspected in the run report.

Components:

* ``judge`` — an optional remote LLM/judge score, if an endpoint is configured.
* ``format`` — deterministic plumbing: the completion is well-formed and not
  degenerate (empty, runaway length).
* ``tool`` — deterministic plumbing: any fenced JSON tool calls parse.
* ``anti_hack`` — deterministic guards: length band and low self-repetition.

Only the judge component makes a semantic call; the rest are parser/validator
checks, consistent with the project's "LLM judges, deterministic plumbing" rule.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
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
    """Fraction of fenced JSON blocks that parse; 1.0 when there are none."""
    blocks = _JSON_BLOCK.findall(completion)
    if not blocks:
        return 1.0
    valid = 0
    for block in blocks:
        try:
            json.loads(block)
        except json.JSONDecodeError:
            continue
        valid += 1
    return valid / len(blocks)


def _anti_hack_score(completion: str, prompt: str) -> float:
    """Penalise copying and runaway repetition."""
    text = completion.strip()
    prompt_words = set(prompt.split())
    words = text.split()
    if not words:
        return 0.0
    copied = sum(1 for word in words if word in prompt_words)
    copy_ratio = copied / len(words)
    distinct = len(set(words)) / len(words)
    score = 1.0
    if copy_ratio > 0.9:
        score -= 0.5
    if distinct < 0.2:
        score -= 0.5
    return max(0.0, score)


class RewardComposer:
    """A callable GRPO reward combining judge, format, tool, and anti-hack terms."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        judge_url: str | None = None,
        judge_timeout_s: float = 30.0,
        max_chars: int = 40_000,
    ) -> None:
        """Initialise the composer with weights and an optional judge endpoint."""
        self._weights = weights or {"judge": 1.0, "format": 0.5, "tool": 0.5, "anti_hack": 0.25}
        self._judge_url = judge_url
        self._judge_timeout = judge_timeout_s
        self._max_chars = max_chars

    def _judge(self, prompt: str, completion: str) -> float:
        """Query an optional remote judge endpoint; 0.0 when unconfigured."""
        if not self._judge_url:
            return 0.0
        import httpx

        response = httpx.post(
            self._judge_url,
            json={"prompt": prompt, "completion": completion},
            timeout=self._judge_timeout,
        )
        response.raise_for_status()
        payload = response.json()
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
            total += self._weights.get("anti_hack", 0.0) * _anti_hack_score(completion, prompt)
            if self._judge_url:
                total += self._weights.get("judge", 0.0) * self._judge(prompt, completion)
            scores.append(total)
        return scores


def build_reward(
    weights: dict[str, float], *, judge_url: str | None = None
) -> Callable[..., list[float]]:
    """Return a GRPO reward function for the given weights."""
    return RewardComposer(weights=weights, judge_url=judge_url)
