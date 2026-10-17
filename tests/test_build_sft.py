"""Tests that the SFT builder refuses to emit unusable records.

A record with no trained token contributes no gradient, and the fallback
tokenizer produces ids unrelated to any real model. Both must be caught rather
than silently written into a dataset.
"""

from __future__ import annotations

from frictiontune.domain.dataset import TrainingExample
from frictiontune.domain.trace import Message, ReasoningPart, TextPart
from frictiontune.sft.tokenizer import FallbackTokenizer
from frictiontune.stages.build_sft import BuildSftSettings, build_record


def test_build_record_flags_a_reasoning_only_completion_as_zero_supervision() -> None:
    example = TrainingExample(
        id="e1",
        source="unit",
        track="recovery",
        context=[
            Message(
                id="m0",
                session_id="s",
                seq=0,
                role="user",
                time_created=0,
                parts=[TextPart(text="hello")],
            )
        ],
        completion=[
            Message(
                id="m1",
                session_id="s",
                seq=1,
                role="assistant",
                time_created=0,
                parts=[ReasoningPart(text="silent thinking")],
            )
        ],
    )

    record, truncated, trained = build_record(
        example,
        FallbackTokenizer(),
        settings=BuildSftSettings(allow_fallback_tokenizer=True),
        tokenizer_kind="fallback",
        max_seq_len=8192,
    )

    assert trained == 0
    assert not truncated
    assert record.meta["tokenizer"] == "fallback"
    assert record.meta["trained_tokens"] == 0
