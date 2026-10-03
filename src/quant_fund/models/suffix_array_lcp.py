"""Suffix array via prefix doubling + Kasai LCP + pattern search.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 914


def _common(s1: str, s2: str) -> int:
    h = 0
    while h < len(s1) and h < len(s2) and s1[h] == s2[h]:
        h += 1
    return h


def suffix_array(s: str) -> list[int]:
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    k = 1
    tmp = [0] * n
    while k < n:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            a, b = sa[i - 1], sa[i]
            pa = (rank[a], rank[a + k] if a + k < n else -1)
            pb = (rank[b], rank[b + k] if b + k < n else -1)
            tmp[b] = tmp[a] + (1 if pb > pa else 0)
        rank = tmp[:]
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


def kasai_lcp(s: str, sa: list[int]) -> list[int]:
    n = len(s)
    inv = [0] * n
    for i, p in enumerate(sa):
        inv[p] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = inv[i]
        if r == 0:
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[r] = h
        h = max(0, h - 1)
    return lcp


def pattern_bounds(s: str, sa: list[int], pat: str) -> tuple[int, int]:
    """[lo,hi) range in sa of suffixes starting with pat."""
    n = len(s)
    m = len(pat)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s[sa[mid] : sa[mid] + m] < pat:
            lo = mid + 1
        else:
            hi = mid
    left = lo
    lo, hi = left, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s[sa[mid] : sa[mid] + m] <= pat:
            lo = mid + 1
        else:
            hi = mid
    return (left, lo)


def bench_suffix_array_lcp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    alpha = "ab"
    s = "".join(rng.choice(list(alpha), 300))
    sa = suffix_array(s)
    truth = sorted(range(len(s)), key=lambda i: s[i:])
    score += 1.0 if sa == truth else 0.0
    lcp = kasai_lcp(s, sa)
    ok = all(lcp[i] == _common(s[sa[i - 1] :], s[sa[i] :]) for i in range(1, len(s)))
    score += 1.0 if ok else 0.0
    # pattern search finds all occurrences
    pat = "aba"
    lo, hi = pattern_bounds(s, sa, pat)
    occ = sorted(sa[i] for i in range(lo, hi))
    truth_occ = [i for i in range(len(s) - len(pat) + 1) if s[i : i + len(pat)] == pat]
    score += 1.0 if occ == truth_occ else 0.0
    # longest repeated substring = max lcp
    lrs_len = max(lcp)
    lrs_truth = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            h = 0
            while i + h < len(s) and j + h < len(s) and s[i + h] == s[j + h]:
                h += 1
            lrs_truth = max(lrs_truth, h)
    score += 1.0 if lrs_len == lrs_truth else 0.0
    return {"synthetic_suffix_array_lcp": score / 4.0}
