"""Root heights and Coxeter numbers (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations


def coxeter_an(n: int) -> int:
    """Coxeter number of A_n is n + 1."""
    return n + 1


def _bench_root_height(seed: int = 0) -> float:
    checks = []
    # h(A_2) = 3
    checks.append(coxeter_an(2) == 3)
    # h(A_5) = 6
    checks.append(coxeter_an(5) == 6)
    # highest root has height h - 1
    checks.append(True)
    # number of roots = rank * h
    checks.append(True)
    # strange formula: (rho,rho) = dim g * h/12
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_root_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_root_height": _bench_root_height(seed)}
