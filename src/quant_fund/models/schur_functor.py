"""Schur functors: Weyl modules S_lambda V (SYNTHETIC)."""

from __future__ import annotations

from math import comb


def sym_dim(d: int, n: int) -> int:
    """dim Sym^d(k^n) = C(n+d-1, d)."""
    return comb(n + d - 1, d)


def wedge_dim(d: int, n: int) -> int:
    """dim wedge^d(k^n) = C(n, d)."""
    return comb(n, d)


def _bench_schur_functor(seed: int = 0) -> float:
    checks = []
    # S_(2) = Sym^2: dim C(n+1,2)
    checks.append(sym_dim(2, 3) == 6)
    # S_(1,1) = wedge^2: dim C(n,2)
    checks.append(wedge_dim(2, 4) == 6)
    # Sym^1 = V
    checks.append(sym_dim(1, 5) == 5)
    # wedge^n = determinant, dim 1
    checks.append(wedge_dim(3, 3) == 1)
    # Sym^2 + wedge^2 = V x V dims
    checks.append(sym_dim(2, 3) + wedge_dim(2, 3) == 9)
    return float(sum(checks) / len(checks))


def bench_schur_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_functor": _bench_schur_functor(seed)}
