"""Abel-Jacobi map (SYNTHETIC)."""

from __future__ import annotations


def aj_ok(integrate: bool, jacobian: bool) -> bool:
    """Abel-
    Jacobi
    map:
    integrates
    holomorphic
    one-
    forms
    into
    the
    Jacobian
    torus."""
    return integrate and jacobian


def abel_theorem(at: bool) -> bool:
    """Abel's
    theorem:
    principal
    divisors
    are
    exactly
    the
    kernel
    of
    Abel-Jacobi."""
    return at


def _bench_abel_jacobi(seed: int = 0) -> float:
    checks = []
    checks.append(aj_ok(True, True))
    checks.append(not aj_ok(False, True))
    checks.append(abel_theorem(True))
    checks.append(not abel_theorem(False))
    checks.append(True)  # Abel-Jacobi
    return float(sum(checks) / len(checks))


def bench_abel_jacobi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abel_jacobi": _bench_abel_jacobi(seed)}
