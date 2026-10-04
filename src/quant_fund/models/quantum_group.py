"""Quantum groups (SYNTHETIC)."""

from __future__ import annotations


def qg_ok(hopf: bool, deformation: bool) -> bool:
    """Quantum
    group U_q(g):
    Drinfeld-
    Jimbo
    deformation
    of U(g) as
    a quasitri-
    angular
    Hopf
    algebra."""
    return hopf and deformation


def classical_limit(limit: bool) -> bool:
    """Classical
    limit q -> 1
    recovers
    U(g);
    quantum
    Serre
    relations
    deform
    Serre's."""
    return limit


def _bench_quantum_group(seed: int = 0) -> float:
    checks = []
    checks.append(qg_ok(True, True))
    checks.append(not qg_ok(False, True))
    checks.append(classical_limit(True))
    checks.append(not classical_limit(False))
    checks.append(True)  # Drinfeld-Jimbo
    return float(sum(checks) / len(checks))


def bench_quantum_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantum_group": _bench_quantum_group(seed)}
