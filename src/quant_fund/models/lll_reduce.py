"""LLL lattice basis reduction over QQ (synthetic) (SYNTHETIC).

Classic LLL with exact Fraction Gram-Schmidt: size reduction and
Lovász condition with delta = 3/4. Verified by (i) the Lovász
conditions holding on output and (ii) first vector within the
provable 2^{(n-1)/2} factor of the true shortest vector (brute-force
over small coefficient box).
"""

from __future__ import annotations

from fractions import Fraction


def _dot(a: list[Fraction], b: list[Fraction]) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _gram_schmidt(
    b: list[list[Fraction]],
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    n = len(b)
    bs: list[list[Fraction]] = [[Fraction(0)] * len(b[0]) for _ in range(n)]
    mu: list[list[Fraction]] = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        v = b[i][:]
        for j in range(i):
            if _dot(bs[j], bs[j]) != 0:
                mu[i][j] = _dot(b[i], bs[j]) / _dot(bs[j], bs[j])
                for k in range(len(v)):
                    v[k] -= mu[i][j] * bs[j][k]
        bs[i] = v
    return bs, mu


def lll(basis: list[list[int]], delta: Fraction = Fraction(3, 4)) -> list[list[Fraction]]:
    b: list[list[Fraction]] = [[Fraction(x) for x in row] for row in basis]
    n = len(b)
    bs, mu = _gram_schmidt(b)
    k = 1
    while k < n:
        # size-reduce b_k
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q != 0:
                for i in range(len(b[0])):
                    b[k][i] -= Fraction(q) * b[j][i]
                bs, mu = _gram_schmidt(b)
        nrm = _dot(bs[k], bs[k])
        nrm_prev = _dot(bs[k - 1], bs[k - 1])
        if nrm >= (delta - mu[k][k - 1] ** 2) * nrm_prev:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            bs, mu = _gram_schmidt(b)
            k = max(k - 1, 1)
    return b


def _norm2(v: list[Fraction]) -> float:
    return float(sum(x * x for x in v))


def _lovasz_ok(b: list[list[Fraction]], delta: Fraction = Fraction(3, 4)) -> bool:
    bs, mu = _gram_schmidt(b)
    n = len(b)
    for k in range(1, n):
        if _dot(bs[k], bs[k]) < (delta - mu[k][k - 1] ** 2) * _dot(bs[k - 1], bs[k - 1]) - Fraction(
            1, 10**9
        ):
            return False
        for j in range(k):
            if abs(mu[k][j]) > Fraction(1, 2) + Fraction(1, 10**9):
                return False
    return True


def bench_lll_reduce(seed: int = 20261231 + 234) -> dict[str, float]:
    # classic nearly-dependent basis
    basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    red = lll(basis)
    ok_lovasz = _lovasz_ok(red)
    # first vector length vs brute-force shortest (small search)
    import itertools

    best = float("inf")
    for coeffs in itertools.product(range(-3, 4), repeat=3):
        if all(c == 0 for c in coeffs):
            continue
        v = [
            sum(
                (Fraction(coeffs[i]) * Fraction(basis[i][j]) for i in range(3)),
                Fraction(0),
            )
            for j in range(3)
        ]
        best = min(best, _norm2(v))
    l1 = _norm2(red[0])
    # LLL bound: ||b1|| <= 2^{(n-1)/2} * lambda1 → in norm²: <= 2^{n-1} * best
    within = l1 <= 4.0 * best + 1e-9
    # gcd-recovery: LLL on [1, a] [0, M] style lattice recovers small vector
    m_basis = [[1, 0, 1000], [0, 1, 517], [0, 0, 1]]
    red2 = lll(m_basis)
    ok2 = _lovasz_ok(red2)
    return {
        "synthetic_lovasz": float(ok_lovasz),
        "synthetic_first_norm": float(l1),
        "synthetic_shortest_norm": float(best),
        "synthetic_within_bound": float(within),
        "synthetic_lovasz2": float(ok2),
        "synthetic_improved": float(l1 < _norm2([Fraction(x) for x in basis[0]])),
    }
