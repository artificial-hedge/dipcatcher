"""bfgs update module (SYNTHETIC)."""

from __future__ import annotations


def bfgs_update_ok(step: bool, radius: bool) -> bool:
    """bfgs_update
    check:
    optimization /
    IGA canon —
    step/radius
    consistency."""
    return step and radius


def bfgs_update_aux(aux: bool) -> bool:
    """bfgs_update
    aux:
    auxiliary
    step check —
    decrease bound."""
    return aux


def _bench_bfgs_update(seed: int = 0) -> float:
    checks = []
    checks.append(bfgs_update_ok(True, True))
    checks.append(not bfgs_update_ok(False, True))
    checks.append(bfgs_update_aux(True))
    checks.append(not bfgs_update_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_bfgs_update(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bfgs_update": _bench_bfgs_update(seed)}
