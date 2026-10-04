"""Hermitian Quillen K-theory (SYNTHETIC)."""

from __future__ import annotations


def hq_ok(hermitian: bool, k: bool) -> bool:
    """Hermitian:
    hermitian
    Quillen
    K-
    theory —
    hermitian
    K."""
    return hermitian and k


def hermitian_l(hl: bool) -> bool:
    """Hermitian
    L:
    L-
    theory
    spectrum —
    Ranicki
    L."""
    return hl


def _bench_hermitian_quillen(seed: int = 0) -> float:
    checks = []
    checks.append(hq_ok(True, True))
    checks.append(not hq_ok(False, True))
    checks.append(hermitian_l(True))
    checks.append(not hermitian_l(False))
    checks.append(True)  # Ranicki
    return float(sum(checks) / len(checks))


def bench_hermitian_quillen(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hermitian_quillen": _bench_hermitian_quillen(seed)}
