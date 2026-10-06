"""Text plumbing: normalisation and n-grams.

These are deterministic transforms used by decontamination (T1/T2) and
reporting. None of them make a semantic judgment.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Produce a case- and punctuation-insensitive form for T2 matching.

    Unicode NFKC folding, lowercasing, punctuation removal, and whitespace
    collapse. This is syntactic normalisation only; it does not paraphrase.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    stripped = _NON_WORD.sub(" ", folded)
    return collapse_whitespace(stripped)


def iter_word_ngrams(text: str, n: int) -> Iterator[str]:
    """Yield overlapping word n-grams of the normalised text (T1 basis)."""
    if n <= 0:
        raise ValueError("n must be positive")
    words = normalize_for_match(text).split()
    for index in range(len(words) - n + 1):
        yield " ".join(words[index : index + n])


def word_ngram_set(text: str, n: int) -> set[str]:
    """Return the set of word n-grams for exact-overlap detection."""
    return set(iter_word_ngrams(text, n))
