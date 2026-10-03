"""Booth's O(n) lexicographically-minimum string rotation.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 917


def booth(s: str) -> int:
    """Index of the least rotation of s."""
    s2 = s + s
    n = len(s)
    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        a, b = s2[i + k], s2[j + k]
        if a == b:
            k += 1
            continue
        if a > b:
            i = i + k + 1
            if i <= j:
                i = j + 1
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
        k = 0
    return min(i, j)


def bench_booth_rotation(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    s = "".join(rng.choice(list("abcd"), 80))
    r = booth(s)
    rot = s[r:] + s[:r]
    truth = min(s[i:] + s[:i] for i in range(len(s)))
    score += 1.0 if rot == truth else 0.0
    # canonical form is rotation-invariant
    t = s[13:] + s[:13]
    rt = booth(t)
    ok2 = (t[rt:] + t[:rt]) == truth
    t2 = s[40:] + s[:40]
    rt2 = booth(t2)
    ok2 = ok2 and (t2[rt2:] + t2[:rt2]) == truth
    score += 1.0 if ok2 else 0.0
    # uniform string → index 0
    score += 1.0 if booth("aaaa") == 0 else 0.0
    # necklace of 'bbaa' → 'aabb' at index 2
    score += 1.0 if booth("bbaa") == 2 else 0.0
    return {"synthetic_booth_rotation": score / 4.0}
