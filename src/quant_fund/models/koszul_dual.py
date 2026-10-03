"""Koszul duality: exterior <-> symmetric algebra (SYNTHETIC)."""

from __future__ import annotations


def ext_exterior(i: int, n: int) -> int:
    """Ext^i_{Lambda(V)}(k, k) = Sym^i(V*): dimension C(n+i-1, i)."""
    from math import comb

    return comb(n + i - 1, i)


def _bench_koszul_dual(seed: int = 0) -> float:
    checks = []
    # Ext^1 = V*, dim n
    checks.append(ext_exterior(1, 3) == 3)
    # Ext^0 = k
    checks.append(ext_exterior(0, 5) == 1)
    # Hilbert series of Ext = (1-t)^{-n} for exterior on n gens
    checks.append(ext_exterior(2, 2) == 3)
    # Koszul dual of Sym(V) is Lambda(V*): symmetric flip
    checks.append(ext_exterior(2, 3) == 6)
    # Poincare duality of the pair: dims symmetric
    checks.append(ext_exterior(1, 4) == 4)
    return float(sum(checks) / len(checks))


def bench_koszul_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koszul_dual": _bench_koszul_dual(seed)}
