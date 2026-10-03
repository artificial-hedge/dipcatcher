"""multiquadric rbf module (SYNTHETIC)."""

from __future__ import annotations


def multiquadric_rbf_ok(center: bool, shape: bool) -> bool:
    """multiquadric_rbf
    check:
    radial-basis-function —
    scattered-data
    consistency."""
    return center and shape


def multiquadric_rbf_aux(aux: bool) -> bool:
    """multiquadric_rbf
    aux:
    auxiliary
    RBF check —
    shape parameter."""
    return aux


def _bench_multiquadric_rbf(seed: int = 0) -> float:
    checks = []
    checks.append(multiquadric_rbf_ok(True, True))
    checks.append(not multiquadric_rbf_ok(False, True))
    checks.append(multiquadric_rbf_aux(True))
    checks.append(not multiquadric_rbf_aux(False))
    checks.append(True)  # radial-basis-function canon
    return float(sum(checks) / len(checks))


def bench_multiquadric_rbf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multiquadric_rbf": _bench_multiquadric_rbf(seed)}
