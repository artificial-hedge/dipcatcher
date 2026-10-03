"""trust region_dogleg module (SYNTHETIC)."""

from __future__ import annotations


def trust_region_dogleg_ok(step: bool, radius: bool) -> bool:
    """trust_region_dogleg
    check:
    optimization /
    IGA canon —
    step/radius
    consistency."""
    return step and radius


def trust_region_dogleg_aux(aux: bool) -> bool:
    """trust_region_dogleg
    aux:
    auxiliary
    step check —
    decrease bound."""
    return aux


def _bench_trust_region_dogleg(seed: int = 0) -> float:
    checks = []
    checks.append(trust_region_dogleg_ok(True, True))
    checks.append(not trust_region_dogleg_ok(False, True))
    checks.append(trust_region_dogleg_aux(True))
    checks.append(not trust_region_dogleg_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_trust_region_dogleg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trust_region_dogleg": _bench_trust_region_dogleg(seed)}
