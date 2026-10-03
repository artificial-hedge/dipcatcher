"""germ grain module (SYNTHETIC)."""

from __future__ import annotations


def germ_grain_ok(geo: bool, tess: bool) -> bool:
    """germ_grain
    check:
    stochastic
    geometry —
    tessellation."""
    return geo and tess


def germ_grain_aux(aux: bool) -> bool:
    """germ_grain
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_germ_grain(seed: int = 0) -> float:
    checks = []
    checks.append(germ_grain_ok(True, True))
    checks.append(not germ_grain_ok(False, True))
    checks.append(germ_grain_aux(True))
    checks.append(not germ_grain_aux(False))
    checks.append(True)  # stochastic-geometry canon
    return float(sum(checks) / len(checks))


def bench_germ_grain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_germ_grain": _bench_germ_grain(seed)}
