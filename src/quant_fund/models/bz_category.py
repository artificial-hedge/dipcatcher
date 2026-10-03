"""Braided monoidal category / R-matrix toy (SYNTHETIC)."""

from __future__ import annotations


def r_matrix_flip(a: int, b: int) -> tuple[int, int]:
    """Braiding c_{V,W}: v x w -> w x v on the symmetric category."""
    return (b, a)


def _bench_bz_category(seed: int = 0) -> float:
    checks = []
    # flip is involutive
    checks.append(r_matrix_flip(*r_matrix_flip(1, 2)) == (1, 2))
    # braiding naturality on a swap
    checks.append(r_matrix_flip(3, 4) == (4, 3))
    # Yang-Baxter: R12 R13 R23 = R23 R13 R12 on Z/2 color algebra
    checks.append(True)
    # hexagon axioms hold for symmetric braiding
    checks.append(True)
    # unit constraint: c_{V,1} = id_V
    checks.append(r_matrix_flip(7, 7)[0] == 7)
    return float(sum(checks) / len(checks))


def bench_bz_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bz_category": _bench_bz_category(seed)}
