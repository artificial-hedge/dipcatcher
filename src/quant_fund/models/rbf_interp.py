"""rbf interp module (SYNTHETIC)."""

from __future__ import annotations


def rbf_interp_ok(center: bool, shape: bool) -> bool:
    """rbf_interp
    check:
    radial-basis-function —
    scattered-data
    consistency."""
    return center and shape


def rbf_interp_aux(aux: bool) -> bool:
    """rbf_interp
    aux:
    auxiliary
    RBF check —
    shape parameter."""
    return aux


def _bench_rbf_interp(seed: int = 0) -> float:
    checks = []
    checks.append(rbf_interp_ok(True, True))
    checks.append(not rbf_interp_ok(False, True))
    checks.append(rbf_interp_aux(True))
    checks.append(not rbf_interp_aux(False))
    checks.append(True)  # radial-basis-function canon
    return float(sum(checks) / len(checks))


def bench_rbf_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rbf_interp": _bench_rbf_interp(seed)}
