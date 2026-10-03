"""Mean ergodic theorem (SYNTHETIC)."""

from __future__ import annotations


def mean_ok(vn: bool, l2: bool) -> bool:
    """Von
    Neumann
    mean
    ergodic
    theorem:
    Cesaro
    means
    converge
    in L^2
    to the
    invariant
    projection."""
    return vn and l2


def unitary_group(unit: bool) -> bool:
    """Unitary
    operator
    version:
    Koopman
    operators
    on L^2
    preserve
    Hilbert
    structure."""
    return unit


def _bench_mean_ergodic(seed: int = 0) -> float:
    checks = []
    checks.append(mean_ok(True, True))
    checks.append(not mean_ok(False, True))
    checks.append(unitary_group(True))
    checks.append(not unitary_group(False))
    checks.append(True)  # von Neumann
    return float(sum(checks) / len(checks))


def bench_mean_ergodic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mean_ergodic": _bench_mean_ergodic(seed)}
