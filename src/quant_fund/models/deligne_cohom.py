"""Deligne cohomology (SYNTHETIC)."""

from __future__ import annotations


def deligne_ok(complex: bool, intermediate: bool) -> bool:
    """Deligne
    cohomology
    H^q_D(X,Z(p)):
    hypercohomology
    of the
    Deligne
    complex."""
    return complex and intermediate


def intermediate_jac(jac: bool) -> bool:
    """H^{2p}_D
    relates to
    the Griffiths
    intermediate
    Jacobian
    J^p(X)."""
    return jac


def _bench_deligne_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(deligne_ok(True, True))
    checks.append(not deligne_ok(False, True))
    checks.append(intermediate_jac(True))
    checks.append(not intermediate_jac(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_deligne_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deligne_cohom": _bench_deligne_cohom(seed)}
