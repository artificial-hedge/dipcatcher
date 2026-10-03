"""SYNTHETIC Smith–Waterman local alignment.

DP local alignment with affine-free scoring (match/mismatch/gap); best score
verified against exhaustive alignment oracle on small strings.
"""

from __future__ import annotations

import random


def smith_waterman(
    a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2
) -> tuple[int, str, str]:
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            H[i][j] = max(
                0,
                H[i - 1][j - 1] + s,
                H[i - 1][j] + gap,
                H[i][j - 1] + gap,
            )
            if H[i][j] > best:
                best, bi, bj = H[i][j], i, j
    # traceback
    aa, bb = [], []
    i, j = bi, bj
    while i > 0 and j > 0 and H[i][j] > 0:
        s = match if a[i - 1] == b[j - 1] else mismatch
        if H[i][j] == H[i - 1][j - 1] + s:
            aa.append(a[i - 1])
            bb.append(b[j - 1])
            i, j = i - 1, j - 1
        elif H[i][j] == H[i - 1][j] + gap:
            aa.append(a[i - 1])
            bb.append("-")
            i -= 1
        else:
            aa.append("-")
            bb.append(b[j - 1])
            j -= 1
    return best, "".join(reversed(aa)), "".join(reversed(bb))


def _score_align(aa: str, bb: str, match: int, mismatch: int, gap: int) -> int:
    s = 0
    for x, y in zip(aa, bb, strict=True):
        if x == "-" or y == "-":
            s += gap
        else:
            s += match if x == y else mismatch
    return s


def _brute_local(a: str, b: str, match: int, mismatch: int, gap: int) -> int:
    """Best local score = best global (linear-gap) over all substring pairs."""
    best = 0
    for i1 in range(len(a) + 1):
        for i2 in range(i1 + 1, len(a) + 1):
            for j1 in range(len(b) + 1):
                for j2 in range(j1 + 1, len(b) + 1):
                    best = max(
                        best,
                        _enum_align_score(a[i1:i2], b[j1:j2], match, mismatch, gap, gap),
                    )
    return best


def _enum_align_score(a: str, b: str, match: int, mismatch: int, go: int, ge: int) -> int:
    """Exact best global alignment score by enumerating all increasing
    matchings (matched-column sets). Independent of any DP."""
    from itertools import combinations

    n, m = len(a), len(b)
    best = -(10**9)
    for k in range(min(n, m) + 1):
        for ai in combinations(range(n), k):
            for bi in combinations(range(m), k):
                s = sum(match if a[ai[t]] == b[bi[t]] else mismatch for t in range(k))
                pa, pb = -1, -1
                runs_a: list[int] = []
                runs_b: list[int] = []
                for t in range(k):
                    runs_a.append(ai[t] - pa - 1)
                    runs_b.append(bi[t] - pb - 1)
                    pa, pb = ai[t], bi[t]
                runs_a.append(n - 1 - pa)
                runs_b.append(m - 1 - pb)
                s += sum(go + (L - 1) * ge for L in runs_a + runs_b if L > 0)
                best = max(best, s)
    return best


def bench_smith_waterman(seed: int = 20261231 + 510) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "ACGT"
    exact = 0
    n = 20
    for _ in range(n):
        a = "".join(rng.choice(alpha) for _ in range(rng.randrange(2, 7)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randrange(2, 7)))
        s, _, _ = smith_waterman(a, b)
        exact += int(s == _brute_local(a, b, 2, -1, -2))
    sane = 0
    for _ in range(n):
        a = "".join(rng.choice(alpha) for _ in range(8))
        b = "".join(rng.choice(alpha) for _ in range(8))
        s, aa, bb = smith_waterman(a, b)
        sane += int(_score_align(aa, bb, 2, -1, -2) == s and len(aa) == len(bb))
    return {
        "synthetic_score_exact": exact / n,
        "synthetic_traceback_consistent": sane / n,
    }
