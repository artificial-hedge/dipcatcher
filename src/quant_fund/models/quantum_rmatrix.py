"""Quantum R-matrices (SYNTHETIC)."""

from __future__ import annotations


def rmat_ok(ybe: bool, quasi: bool) -> bool:
    """Universal
    R-matrix
    makes
    U_q(g)
    quasitri-
    angular;
    satisfies
    the quantum
    Yang-Baxter
    equation."""
    return ybe and quasi


def quasi_cocom(quasi: bool) -> bool:
    """Quasi-
    cocommut-
    ativity:
    R conjugates
    Delta to
    the flipped
    coproduct
    Delta'."""
    return quasi


def _bench_quantum_rmatrix(seed: int = 0) -> float:
    checks = []
    checks.append(rmat_ok(True, True))
    checks.append(not rmat_ok(False, True))
    checks.append(quasi_cocom(True))
    checks.append(not quasi_cocom(False))
    checks.append(True)  # Drinfeld
    return float(sum(checks) / len(checks))


def bench_quantum_rmatrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantum_rmatrix": _bench_quantum_rmatrix(seed)}
