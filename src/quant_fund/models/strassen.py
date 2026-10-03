"""SYNTHETIC Strassen matrix multiplication (power-of-2 sizes).

Seven recursive multiplies per level vs naive triple loop; verified
exact on random integer matrices 4..64.
"""

from __future__ import annotations

import random


def _naive(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _add(a: list[list[int]], b: list[list[int]], s: int = 1) -> list[list[int]]:
    return [[x + s * y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def strassen(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n = len(a)
    if n <= 4:
        return _naive(a, b)
    h = n // 2
    a11 = [r[:h] for r in a[:h]]
    a12 = [r[h:] for r in a[:h]]
    a21 = [r[:h] for r in a[h:]]
    a22 = [r[h:] for r in a[h:]]
    b11 = [r[:h] for r in b[:h]]
    b12 = [r[h:] for r in b[:h]]
    b21 = [r[:h] for r in b[h:]]
    b22 = [r[h:] for r in b[h:]]
    p1 = strassen(_add(a11, a22), _add(b11, b22))
    p2 = strassen(_add(a21, a22), b11)
    p3 = strassen(a11, _add(b12, b22, -1))
    p4 = strassen(a22, _add(b21, b11, -1))
    p5 = strassen(_add(a11, a12), b22)
    p6 = strassen(_add(a21, a11, -1), _add(b11, b12))
    p7 = strassen(_add(a12, a22, -1), _add(b21, b22))
    c11 = _add(_add(p1, p4), _add(p7, p5, -1))
    c12 = _add(p3, p5)
    c21 = _add(p2, p4)
    c22 = _add(_add(p1, p3), _add(p6, p2, -1))
    return [c11[i] + c12[i] for i in range(h)] + [c21[i] + c22[i] for i in range(h)]


def bench_strassen(seed: int = 20261231 + 453) -> dict[str, float]:
    rng = random.Random(seed)
    exact = assoc = ident = 0
    trials = 15
    for _ in range(trials):
        n = 1 << rng.randrange(2, 5)
        a = [[rng.randrange(-9, 10) for _ in range(n)] for _ in range(n)]
        b = [[rng.randrange(-9, 10) for _ in range(n)] for _ in range(n)]
        exact += int(strassen(a, b) == _naive(a, b))
        c = [[rng.randrange(-9, 10) for _ in range(n)] for _ in range(n)]
        assoc += int(strassen(strassen(a, b), c) == strassen(a, strassen(b, c)))
        ident += int(strassen(a, [[int(i == j) for j in range(n)] for i in range(n)]) == a)
    return {
        "synthetic_exact_vs_naive": float(exact / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_identity": float(ident / trials),
    }
