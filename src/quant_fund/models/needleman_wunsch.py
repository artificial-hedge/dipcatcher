"""SYNTHETIC Needleman–Wunsch global alignment with affine gap penalties.

Gotoh three-matrix DP (open/extend); score + end-gap semantics verified
against exhaustive enumeration on short strings.
"""

from __future__ import annotations

import random


def nw_affine(
    a: str, b: str, match: int = 2, mismatch: int = -1, go: int = -3, ge: int = -1
) -> tuple[int, str, str]:
    n, m = len(a), len(b)
    NEG = -(10**9)
    M = [[NEG] * (m + 1) for _ in range(n + 1)]
    X = [[NEG] * (m + 1) for _ in range(n + 1)]  # gap in b (vertical)
    Y = [[NEG] * (m + 1) for _ in range(n + 1)]  # gap in a (horizontal)
    M[0][0] = 0
    for i in range(1, n + 1):
        X[i][0] = go + (i - 1) * ge
    for j in range(1, m + 1):
        Y[0][j] = go + (j - 1) * ge
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + s
            X[i][j] = max(M[i - 1][j] + go, X[i - 1][j] + ge, Y[i - 1][j] + go)
            Y[i][j] = max(M[i][j - 1] + go, Y[i][j - 1] + ge, X[i][j - 1] + go)
    best = max(M[n][m], X[n][m], Y[n][m])
    # traceback
    aa, bb = [], []
    i, j, state = n, m, ("M" if M[n][m] == best else ("X" if X[n][m] == best else "Y"))
    while i > 0 or j > 0:
        if state == "M":
            aa.append(a[i - 1])
            bb.append(b[j - 1])
            cur = M[i][j]
            s = match if a[i - 1] == b[j - 1] else mismatch
            if cur - s == M[i - 1][j - 1]:
                state = "M"
            elif cur - s == X[i - 1][j - 1]:
                state = "X"
            else:
                state = "Y"
            i, j = i - 1, j - 1
        elif state == "X":
            aa.append(a[i - 1])
            bb.append("-")
            cur = X[i][j]
            if cur - go == M[i - 1][j]:
                state = "M"
            elif cur - ge == X[i - 1][j]:
                state = "X"
            else:
                state = "Y"
            i -= 1
        else:
            aa.append("-")
            bb.append(b[j - 1])
            cur = Y[i][j]
            if cur - go == M[i][j - 1]:
                state = "M"
            elif cur - ge == Y[i][j - 1]:
                state = "Y"
            else:
                state = "X"
            j -= 1
    return best, "".join(reversed(aa)), "".join(reversed(bb))


def _affine_score(aa: str, bb: str, match: int, mismatch: int, go: int, ge: int) -> int:
    s = 0
    i = 0
    while i < len(aa):
        if aa[i] == "-":
            j = i
            while j < len(aa) and aa[j] == "-":
                j += 1
            s += go + (j - i - 1) * ge
            i = j
        elif bb[i] == "-":
            j = i
            while j < len(bb) and bb[j] == "-":
                j += 1
            s += go + (j - i - 1) * ge
            i = j
        else:
            s += match if aa[i] == bb[i] else mismatch
            i += 1
    return s


def bench_needleman_wunsch(seed: int = 20261231 + 511) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "ACG"
    consistent = 0
    n = 40
    for _ in range(n):
        a = "".join(rng.choice(alpha) for _ in range(rng.randrange(2, 8)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randrange(2, 8)))
        s, aa, bb = nw_affine(a, b)
        consistent += int(
            len(aa) == len(bb)
            and _affine_score(aa, bb, 2, -1, -3, -1) == s
            and aa.replace("-", "") == a
            and bb.replace("-", "") == b
        )
    # exact oracle on tiny strings: enumerate all alignments
    exact = 0
    n2 = 25
    for _ in range(n2):
        a = "".join(rng.choice(alpha) for _ in range(rng.randrange(1, 5)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randrange(1, 5)))
        s, _, _ = nw_affine(a, b)
        exact += int(s == _brute_global(a, b, 2, -1, -3, -1))
    return {
        "synthetic_traceback_consistent": consistent / n,
        "synthetic_score_exact": exact / n2,
    }


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


def _brute_global(a: str, b: str, match: int, mismatch: int, go: int, ge: int) -> int:
    return _enum_align_score(a, b, match, mismatch, go, ge)
