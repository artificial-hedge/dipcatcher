"""Duval's Lyndon factorization: split s into non-increasing Lyndon words.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 918


def duval(s: str) -> list[str]:
    n = len(s)
    out: list[str] = []
    i = 0
    while i < n:
        j, k = i + 1, i
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        m = j - k
        while i <= k:
            out.append(s[i : i + m])
            i += m
    return out


def _is_lyndon(w: str) -> bool:
    return all(w < w[i:] + w[:i] for i in range(1, len(w)))


def bench_lyndon_factor(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    s = "".join(rng.choice(list("abc"), 100))
    parts = duval(s)
    # concatenation reconstructs s
    score += 1.0 if "".join(parts) == s else 0.0
    # each factor is Lyndon
    score += 1.0 if all(_is_lyndon(p) for p in parts) else 0.0
    # factors non-increasing lexicographically
    score += 1.0 if all(parts[i] >= parts[i + 1] for i in range(len(parts) - 1)) else 0.0
    # single Lyndon word → one factor
    score += 1.0 if duval("aab") == ["aab"] and duval("cba") == ["c", "b", "a"] else 0.0
    return {"synthetic_lyndon_factor": score / 4.0}
