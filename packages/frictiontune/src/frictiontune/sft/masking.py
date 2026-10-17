"""Token-level masking and context-window fitting.

Labels use ``-100`` (the PyTorch ignore index) on every non-trained token so the
loss is computed only over the assistant behaviour we want to teach. Fitting
keeps the trained span and as much surrounding context as the window allows,
dropping the oldest context and any trailing tokens that do not fit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from frictiontune.sft.renderer import RenderedChat

IGNORE_INDEX = -100


class FittedExample(BaseModel):
    """A rendered example plus its training labels, fitted to a window."""

    model_config = ConfigDict(extra="forbid")

    input_ids: list[int]
    labels: list[int]
    truncated: bool


def build_labels(rendered: RenderedChat, *, ignore_index: int = IGNORE_INDEX) -> list[int]:
    """Return labels where non-trained tokens are masked with ``ignore_index``."""
    rendered.check()
    return [
        token if trained else ignore_index
        for token, trained in zip(rendered.input_ids, rendered.assistant_mask, strict=True)
    ]


def fit_window(rendered: RenderedChat, max_seq_len: int) -> RenderedChat:
    """Trim a rendered chat to ``max_seq_len`` tokens, preserving the trained span.

    The trained span always survives. If the span alone is longer than the
    window its tail is kept (the end of the behaviour is what matters most);
    otherwise the window is placed to keep the span with as much preceding
    context as fits, dropping whichever of the oldest context or the trailing
    tokens do not fit.
    """
    length = len(rendered.input_ids)
    if length <= max_seq_len:
        return rendered

    trained = [index for index, flag in enumerate(rendered.assistant_mask) if flag]
    if not trained:
        start = length - max_seq_len
        return RenderedChat(
            input_ids=rendered.input_ids[start:],
            assistant_mask=rendered.assistant_mask[start:],
        )

    first, last = trained[0], trained[-1]
    span = last - first + 1
    if span >= max_seq_len:
        start = last - max_seq_len + 1
        end = last + 1
    else:
        context_budget = max_seq_len - span
        start = max(0, first - context_budget)
        end = min(length, start + max_seq_len)
        start = max(0, end - max_seq_len)
    return RenderedChat(
        input_ids=rendered.input_ids[start:end],
        assistant_mask=rendered.assistant_mask[start:end],
    )


def fit_and_label(rendered: RenderedChat, max_seq_len: int) -> FittedExample:
    """Fit a rendered chat to the window and produce its labels."""
    fitted = fit_window(rendered, max_seq_len)
    return FittedExample(
        input_ids=fitted.input_ids,
        labels=build_labels(fitted),
        truncated=len(fitted.input_ids) < len(rendered.input_ids),
    )
