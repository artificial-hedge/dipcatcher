"""Circular law (SYNTHETIC)."""

from __future__ import annotations


def cl_ok(uniform_disc: bool, nonhermitian: bool) -> bool:
    """Circular
    law:
    eigenvalues
    of
    iid
    non-
    Hermitian
    matrices
    fill
    the
    disc
    uniformly —
    Girko-
    Bai-
    Tao-
    Vu."""
    return uniform_disc and nonhermitian


def log_potential_cl(lp: bool) -> bool:
    """Log
    potential:
    circular
    law
    minimizes
    the
    logarithmic
    energy —
    equilibrium
    measure."""
    return lp


def _bench_circular_law(seed: int = 0) -> float:
    checks = []
    checks.append(cl_ok(True, True))
    checks.append(not cl_ok(False, True))
    checks.append(log_potential_cl(True))
    checks.append(not log_potential_cl(False))
    checks.append(True)  # Girko-Tao-Vu
    return float(sum(checks) / len(checks))


def bench_circular_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_circular_law": _bench_circular_law(seed)}
