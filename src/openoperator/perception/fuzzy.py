"""
Fuzzy matching utilities for OCR text matching.

Provides lightweight Levenshtein distance and token-set matching
implementations so the project doesn't require heavy external deps.
"""
from __future__ import annotations

from typing import Iterable


def levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein distance between two strings.

    Time complexity O(len(a) * len(b)).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # ensure a is the shorter string to use less memory
    if len(a) > len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def similarity_ratio(a: str, b: str) -> float:
    """Return a similarity ratio between 0.0 and 1.0 based on Levenshtein.

    If both strings are empty returns 1.0.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - (dist / max_len))


def _tokenize(text: str) -> list[str]:
    return [t for t in (text or "").lower().split() if t]


def token_set_ratio(a: str, b: str) -> float:
    """Compute a token-set based similarity.

    This is a lightweight approximation inspired by fuzzywuzzy's token_set_ratio:
    - Compare intersection tokens as a joined string against the union/remaining tokens
    - Return the best similarity ratio among those comparisons.
    """
    toks_a = _tokenize(a)
    toks_b = _tokenize(b)
    set_a = set(toks_a)
    set_b = set(toks_b)

    if not set_a and not set_b:
        return 1.0

    common = set_a & set_b
    diff_a = list(set_a - common)
    diff_b = list(set_b - common)

    # Build comparison strings
    common_str = " ".join(sorted(common))
    a_rem = " ".join(sorted(diff_a))
    b_rem = " ".join(sorted(diff_b))

    candidates: list[tuple[str, str]] = []
    if common_str:
        candidates.append((common_str, common_str))
    if common_str and a_rem:
        candidates.append((common_str, a_rem))
    if common_str and b_rem:
        candidates.append((common_str, b_rem))
    if a_rem and b_rem:
        candidates.append((a_rem, b_rem))
    if not candidates:
        # fallback to direct comparison
        candidates.append((a, b))

    best = 0.0
    for x, y in candidates:
        best = max(best, similarity_ratio(x, y))
    return best


def fuzzy_match_score(a: str, b: str) -> float:
    """Return a combined fuzzy score between 0.0 and 1.0.

    Uses both simple similarity ratio and token_set_ratio and returns the
    maximum value so callers can use a tolerant matching threshold.
    """
    return max(similarity_ratio(a, b), token_set_ratio(a, b))
