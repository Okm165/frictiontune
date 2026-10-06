"""Deterministic identity helpers for domain objects.

IDs are content-derived so the same input always yields the same id; this makes
runs comparable and artifacts diffable.
"""

from __future__ import annotations

from frictiontune.core.hashing import content_hash, short


def session_id(raw: str) -> str:
    """Return a stable session id from the source identifier."""
    return raw


def message_id(session: str, seq: int) -> str:
    """Return a stable message id within a session."""
    return f"{session}#{seq}"


def episode_id(session: str, start_message_id: str) -> str:
    """Return a stable episode id anchored at the opening user message."""
    return f"{session}:{start_message_id}"


def phase_id(episode: str, index: int) -> str:
    """Return a stable phase id within an episode."""
    return f"{episode}/phase{index}"


def event_id(episode: str, index: int) -> str:
    """Return a stable friction-event id within an episode."""
    return f"{episode}/f{index}"


def rule_id(instruction: str, applicability: str) -> str:
    """Return a content-derived rule id."""
    return (
        f"rule-{short(content_hash({'instruction': instruction, 'applicability': applicability}))}"
    )


def example_id(source: str, track: str, index: int) -> str:
    """Return a stable training-example id."""
    return f"{source}/{track}/{index}"


def candidate_id(source: str, index: int) -> str:
    """Return a stable candidate id for judging."""
    return f"{source}/{index}"
