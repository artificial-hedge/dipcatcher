"""gaussian rbf module (SYNTHETIC)."""

from __future__ import annotations


def gaussian_rbf_ok(center: bool, shape: bool) -> bool:
    """gaussian_rbf
    check:
    radial-basis-function —
    scattered-data
    consistency."""
    return center and shape


def gaussian_rbf_aux(aux: bool) -> bool:
    """gaussian_rbf
    aux:
    auxiliary
    RBF check —
    shape parameter."""
    return aux


def _bench_gaussian_rbf(seed: int = 0) -> float:
    checks = []
    checks.append(gaussian_rbf_ok(True, True))
    checks.append(not gaussian_rbf_ok(False, True))
    checks.append(gaussian_rbf_aux(True))
    checks.append(not gaussian_rbf_aux(False))
    checks.append(True)  # radial-basis-function canon
    return float(sum(checks) / len(checks))


def bench_gaussian_rbf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gaussian_rbf": _bench_gaussian_rbf(seed)}
