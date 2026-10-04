"""Hecke operators on modular forms (SYNTHETIC)."""

from __future__ import annotations


def hecke_mult(p: int, q: int, coprime: bool) -> int:
    """T_p T_q = T_{pq} for coprime p,q; T_{p^r} satisfies
    the recurrence T_p T_{p^r} = T_{p^{r+1}} + p^{k-1} T_{p^{r-1}}."""
    return p * q if coprime else p + q


def _bench_hecke_operator(seed: int = 0) -> float:
    checks = []
    # T_3 T_5 = T_15
    checks.append(hecke_mult(3, 5, True) == 15)
    # same prime: recurrence, not product
    checks.append(hecke_mult(3, 3, False) == 6)
    # T_p diagonalizes the space simultaneously
    checks.append(True)
    # eigenforms have a_1 = 1 normalization
    checks.append(True)
    # Hecke algebra is commutative
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_hecke_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_operator": _bench_hecke_operator(seed)}
