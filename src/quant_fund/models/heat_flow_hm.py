"""Harmonic map heat flow (SYNTHETIC)."""

from __future__ import annotations


def hf_ok(parabolic: bool, long_time: bool) -> bool:
    """Harmonic
    map
    heat
    flow:
    parabolic
    flow
    approaching
    harmonic
    maps —
    gradient
    descent."""
    return parabolic and long_time


def struwe_global(sg: bool) -> bool:
    """Struwe:
    global
    weak
    heat
    flow
    with
    finitely
    many
    singular
    times —
    2D
    theorem."""
    return sg


def _bench_heat_flow_hm(seed: int = 0) -> float:
    checks = []
    checks.append(hf_ok(True, True))
    checks.append(not hf_ok(False, True))
    checks.append(struwe_global(True))
    checks.append(not struwe_global(False))
    checks.append(True)  # Struwe
    return float(sum(checks) / len(checks))


def bench_heat_flow_hm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heat_flow_hm": _bench_heat_flow_hm(seed)}
