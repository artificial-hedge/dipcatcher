"""KMP linear-time string matching (synthetic) (SYNTHETIC).

Prefix-function automaton; verified against str.find oracle incl.
overlapping occurrences and pathological inputs (aaaa…ab).
"""

from __future__ import annotations

import random


def prefix_fn(p: str) -> list[int]:
    pi = [0] * len(p)
    k = 0
    for i in range(1, len(p)):
        while k and p[i] != p[k]:
            k = pi[k - 1]
        if p[i] == p[k]:
            k += 1
        pi[i] = k
    return pi


def kmp_find(text: str, pat: str) -> list[int]:
    if not pat:
        return []
    pi = prefix_fn(pat)
    hits = []
    q = 0
    for i, c in enumerate(text):
        while q and c != pat[q]:
            q = pi[q - 1]
        if c == pat[q]:
            q += 1
        if q == len(pat):
            hits.append(i - len(pat) + 1)
            q = pi[q - 1]
    return hits


def bench_kmp_search(seed: int = 20261231 + 262) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "ab"
    agree = 0
    trials = 50
    for _ in range(trials):
        text = "".join(rng.choice(alpha) for _ in range(rng.randint(20, 300)))
        pat = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 15)))
        want = [i for i in range(len(text) - len(pat) + 1) if text[i : i + len(pat)] == pat]
        agree += int(kmp_find(text, pat) == want)
    # worst case: aaaa…a vs aa…ab → O(n+m)
    text = "a" * 2000 + "b"
    pat = "a" * 15 + "b"
    worst = kmp_find(text, pat) == [len(text) - len(pat)]
    # overlapping: 'aaa' in 'aaaaa' → positions 0,1,2
    overlap = kmp_find("aaaaa", "aaa") == [0, 1, 2]
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_worstcase": float(worst),
        "synthetic_overlap": float(overlap),
    }
