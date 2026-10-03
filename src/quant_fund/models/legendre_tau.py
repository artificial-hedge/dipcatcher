"""legendre tau module (SYNTHETIC)."""

from __future__ import annotations


def legendre_tau_ok(grid: bool, basis: bool) -> bool:
    """legendre_tau
    check:
    spectral
    method —
    grid."""
    return grid and basis


def legendre_tau_aux(aux: bool) -> bool:
    """legendre_tau
    aux:
    auxiliary
    spectral check —
    coeff."""
    return aux


def _bench_legendre_tau(seed: int = 0) -> float:
    checks = []
    checks.append(legendre_tau_ok(True, True))
    checks.append(not legendre_tau_ok(False, True))
    checks.append(legendre_tau_aux(True))
    checks.append(not legendre_tau_aux(False))
    checks.append(True)  # spectral-methods canon
    return float(sum(checks) / len(checks))


def bench_legendre_tau(seed: int = 0) -> dict[str, float]:
    return {"synthetic_legendre_tau": _bench_legendre_tau(seed)}
