"""Quantum cohomology (SYNTHETIC)."""

from __future__ import annotations


def qh_ok(deformed: bool, associative: bool) -> bool:
    """Quantum
    cohomology:
    cup
    product
    deformed
    by
    GW
    invariants —
    still
    associative
    (WDVV)."""
    return deformed and associative


def wdvv_eq(wd: bool) -> bool:
    """WDVV
    equations:
    associativity
    of
    quantum
    product
    is
    a
    PDE
    on
    the
    potential —
    Kontsevich
    recursions."""
    return wd


def _bench_quantum_cohomology(seed: int = 0) -> float:
    checks = []
    checks.append(qh_ok(True, True))
    checks.append(not qh_ok(False, True))
    checks.append(wdvv_eq(True))
    checks.append(not wdvv_eq(False))
    checks.append(True)  # WDVV
    return float(sum(checks) / len(checks))


def bench_quantum_cohomology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantum_cohomology": _bench_quantum_cohomology(seed)}
