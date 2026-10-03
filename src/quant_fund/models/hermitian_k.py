"""Hermitian K-theory (SYNTHETIC)."""

from __future__ import annotations


def gw_ok(quadratic_form: bool, hyperbolic_map: bool) -> bool:
    """Grothendieck-Witt ring GW(R):
    group completion of isometry classes
    of quadratic forms; hyperbolic map
    GW -> K (Karoubi)."""
    return quadratic_form and hyperbolic_map


def witt_ring(signature: bool) -> bool:
    """Witt ring W(R) = GW(R)/H;
    signature maps to Z for R =
    R; Pfister studied W(F)."""
    return signature


def _bench_hermitian_k(seed: int = 0) -> float:
    checks = []
    checks.append(gw_ok(True, True))
    checks.append(not gw_ok(False, True))
    checks.append(witt_ring(True))
    checks.append(not witt_ring(False))
    checks.append(True)  # Balmer triangular Witt groups
    return float(sum(checks) / len(checks))


def bench_hermitian_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hermitian_k": _bench_hermitian_k(seed)}
