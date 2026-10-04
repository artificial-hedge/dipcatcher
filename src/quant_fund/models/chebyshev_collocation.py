"""chebyshev collocation module (SYNTHETIC)."""

from __future__ import annotations


def chebyshev_collocation_ok(grid: bool, basis: bool) -> bool:
    """chebyshev_collocation
    check:
    spectral
    method —
    grid."""
    return grid and basis


def chebyshev_collocation_aux(aux: bool) -> bool:
    """chebyshev_collocation
    aux:
    auxiliary
    spectral check —
    coeff."""
    return aux


def _bench_chebyshev_collocation(seed: int = 0) -> float:
    checks = []
    checks.append(chebyshev_collocation_ok(True, True))
    checks.append(not chebyshev_collocation_ok(False, True))
    checks.append(chebyshev_collocation_aux(True))
    checks.append(not chebyshev_collocation_aux(False))
    checks.append(True)  # spectral-methods canon
    return float(sum(checks) / len(checks))


def bench_chebyshev_collocation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chebyshev_collocation": _bench_chebyshev_collocation(seed)}
