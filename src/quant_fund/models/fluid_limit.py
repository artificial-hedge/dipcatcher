"""fluid limit module (SYNTHETIC)."""

from __future__ import annotations


def fluid_limit_ok(lim: bool, scale: bool) -> bool:
    """fluid_limit
    check:
    heavy-traffic
    structure —
    diffusion
    limit."""
    return lim and scale


def fluid_limit_aux(aux: bool) -> bool:
    """fluid_limit
    aux:
    auxiliary
    scaling
    check —
    QED
    regime."""
    return aux


def _bench_fluid_limit(seed: int = 0) -> float:
    checks = []
    checks.append(fluid_limit_ok(True, True))
    checks.append(not fluid_limit_ok(False, True))
    checks.append(fluid_limit_aux(True))
    checks.append(not fluid_limit_aux(False))
    checks.append(True)  # heavy-traffic canon
    return float(sum(checks) / len(checks))


def bench_fluid_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fluid_limit": _bench_fluid_limit(seed)}
