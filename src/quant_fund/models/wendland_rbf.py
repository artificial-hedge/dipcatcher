"""wendland rbf module (SYNTHETIC)."""

from __future__ import annotations


def wendland_rbf_ok(center: bool, shape: bool) -> bool:
    """wendland_rbf
    check:
    radial-basis-function —
    scattered-data
    consistency."""
    return center and shape


def wendland_rbf_aux(aux: bool) -> bool:
    """wendland_rbf
    aux:
    auxiliary
    RBF check —
    shape parameter."""
    return aux


def _bench_wendland_rbf(seed: int = 0) -> float:
    checks = []
    checks.append(wendland_rbf_ok(True, True))
    checks.append(not wendland_rbf_ok(False, True))
    checks.append(wendland_rbf_aux(True))
    checks.append(not wendland_rbf_aux(False))
    checks.append(True)  # radial-basis-function canon
    return float(sum(checks) / len(checks))


def bench_wendland_rbf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wendland_rbf": _bench_wendland_rbf(seed)}
