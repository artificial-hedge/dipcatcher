"""Bitap (Shift-And) fuzzy string matching.

Finds all positions where `pattern` matches a substring of `text` with at
most k Levenshtein errors (insert/delete/substitute), using a bit-parallel
state per error level. SYNTHETIC bench verifies against an O(nm) DP oracle.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 928


def bitap_search(pattern: str, text: str, k: int) -> list[tuple[int, int]]:
    """Return (end_pos, errors) for every end position matching with <=k."""
    m = len(pattern)
    if m == 0:
        return [(i, 0) for i in range(len(text) + 1)]
    masks: dict[str, int] = {}
    for i, ch in enumerate(pattern):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    # R[j]: bit i set <=> pattern[0..i] matches a suffix of processed text
    # with <= j errors. Init: matching the empty text costs i+1 deletions.
    r = [(1 << j) - 1 for j in range(k + 1)]
    top = 1 << (m - 1)
    out: list[tuple[int, int]] = []
    for ti, ch in enumerate(text):
        cm = masks.get(ch, 0)
        prev = r[:]
        r[0] = ((prev[0] << 1) | 1) & cm
        for j in range(1, k + 1):
            r[j] = (
                (((prev[j] << 1) | 1) & cm)  # match
                | ((r[j - 1] << 1) | 1)  # deletion of pattern char
                | ((prev[j - 1] << 1) | 1)  # substitution
                | prev[j - 1]  # insertion in text
            )
        if r[k] & top:
            out.append((ti + 1, k))
    return out


def _lev_oracle(pattern: str, text: str, k: int) -> set[int]:
    """End positions where pattern matches some substring with <=k edits."""
    m, n = len(pattern), len(text)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        d[0][j] = j
    ends: set[int] = set()
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if text[i - 1] == pattern[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        if d[i][m] <= k:
            ends.add(i)
    return ends


def bench_bitap_fuzzy(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    got = {e for e, _ in bitap_search("needle", "the needle is near the neddle", 1)}
    want = _lev_oracle("needle", "the needle is near the neddle", 1)
    score += 1.0 if got == want and want else 0.0
    alpha = "ab"
    ok = True
    for _ in range(120):
        p = "".join(str(rng.choice(list(alpha))) for _ in range(int(rng.integers(3, 7))))
        t = "".join(str(rng.choice(list(alpha))) for _ in range(int(rng.integers(6, 20))))
        k = int(rng.integers(0, 3))
        got = {e for e, _ in bitap_search(p, t, k)}
        if got != _lev_oracle(p, t, k):
            ok = False
            break
    score += 1.0 if ok else 0.0
    score += 1.0 if {e for e, _ in bitap_search("abc", "zzabczz", 0)} == {5} else 0.0
    score += (
        1.0
        if {e for e, _ in bitap_search("abc", "axc", 1)} == {3}
        and not bitap_search("abc", "axc", 0)
        else 0.0
    )
    return {"synthetic_bitap_fuzzy": score / 4.0}
