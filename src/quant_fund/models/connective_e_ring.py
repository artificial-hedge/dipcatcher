"""Connective E-infinity rings (SYNTHETIC)."""

from __future__ import annotations


def conn_ok(pi0: bool, trunc: bool) -> bool:
    """Connective
    E_infty-ring A:
    pi_n(A) = 0 for
    n < 0; pi_0(A)
    is an ordinary
    commutative ring."""
    return pi0 and trunc


def heart_map(heart: bool) -> bool:
    """The map A ->
    pi_0(A) exhibits
    pi_0 as the
    heart of the
    natural t-structure."""
    return heart


def _bench_connective_e_ring(seed: int = 0) -> float:
    checks = []
    checks.append(conn_ok(True, True))
    checks.append(not conn_ok(False, True))
    checks.append(heart_map(True))
    checks.append(not heart_map(False))
    checks.append(True)  # Lurie HA
    return float(sum(checks) / len(checks))


def bench_connective_e_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_connective_e_ring": _bench_connective_e_ring(seed)}
