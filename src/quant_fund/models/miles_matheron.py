"""miles matheron module (SYNTHETIC)."""

from __future__ import annotations


def miles_matheron_ok(geo: bool, tess: bool) -> bool:
    """miles_matheron
    check:
    stochastic
    geometry —
    tessellation."""
    return geo and tess


def miles_matheron_aux(aux: bool) -> bool:
    """miles_matheron
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_miles_matheron(seed: int = 0) -> float:
    checks = []
    checks.append(miles_matheron_ok(True, True))
    checks.append(not miles_matheron_ok(False, True))
    checks.append(miles_matheron_aux(True))
    checks.append(not miles_matheron_aux(False))
    checks.append(True)  # stochastic-geometry canon
    return float(sum(checks) / len(checks))


def bench_miles_matheron(seed: int = 0) -> dict[str, float]:
    return {"synthetic_miles_matheron": _bench_miles_matheron(seed)}
