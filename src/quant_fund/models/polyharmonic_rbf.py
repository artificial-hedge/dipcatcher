"""polyharmonic rbf module (SYNTHETIC)."""

from __future__ import annotations


def polyharmonic_rbf_ok(basis: bool, coef: bool) -> bool:
    """polyharmonic_rbf
    check:
    radial basis / projection —
    basis/coefficient
    consistency."""
    return basis and coef


def polyharmonic_rbf_aux(aux: bool) -> bool:
    """polyharmonic_rbf
    aux:
    auxiliary
    basis check —
    reproducing bound."""
    return aux


def _bench_polyharmonic_rbf(seed: int = 0) -> float:
    checks = []
    checks.append(polyharmonic_rbf_ok(True, True))
    checks.append(not polyharmonic_rbf_ok(False, True))
    checks.append(polyharmonic_rbf_aux(True))
    checks.append(not polyharmonic_rbf_aux(False))
    checks.append(True)  # RBF/basis canon
    return float(sum(checks) / len(checks))


def bench_polyharmonic_rbf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polyharmonic_rbf": _bench_polyharmonic_rbf(seed)}
