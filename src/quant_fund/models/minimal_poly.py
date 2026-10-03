"""Minimal polynomial degree over GF(p) via power-dependence (SYNTHETIC)."""

from __future__ import annotations

import itertools

from quant_fund.models.field_ext import padd, pdivmod, pmul


def in_span(v: list[int], basis: list[list[int]], p: int) -> bool:
    """Brute-force coefficients in GF(p)."""
    for coefs in itertools.product(range(p), repeat=len(basis)):
        acc: list[int] = [0]
        for c, b in zip(coefs, basis, strict=True):
            acc = padd(acc, [c * x % p for x in b], p)
        if acc == v:
            return True
    return False


def minimal_poly_degree(alpha: list[int], modulus: list[int], p: int, max_deg: int = 8) -> int:
    """Smallest d such that a^d lies in span{1, a, ..., a^{d-1}} mod (modulus)."""
    powers = [[1]]
    cur = [1]
    for _d in range(1, max_deg + 1):
        cur = pdivmod(pmul(cur, alpha, p), modulus, p)[1]
        powers.append(cur)
        if in_span(cur, powers[:-1], p):
            return _d
    return -1


def _bench_minimal_poly(seed: int = 0) -> float:
    checks = []
    # GF(4) = GF(2)[x]/(x^2+x+1): alpha = x has minimal poly deg 2
    checks.append(minimal_poly_degree([0, 1], [1, 1, 1], 2) == 2)
    # element 1 has minimal poly deg 1 (x-1)
    checks.append(minimal_poly_degree([1], [1, 1, 1], 2) == 1)
    # GF(9) = GF(3)[x]/(x^2+1): alpha = x has deg 2
    checks.append(minimal_poly_degree([0, 1], [1, 0, 1], 3) == 2)
    # arbitrary element has degree dividing the extension degree
    checks.append(minimal_poly_degree([1, 1], [1, 0, 1], 3) in (1, 2))
    return float(sum(checks) / len(checks))


def bench_minimal_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minimal_poly": _bench_minimal_poly(seed)}
