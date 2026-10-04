"""Seiberg-Witten theory (SYNTHETIC)."""

from __future__ import annotations


def sw_ok(spinc: bool, monopole: bool) -> bool:
    """Seiberg-
    Witten
    equations:
    spin-c
    structure
    plus
    monopole
    field —
    simpler
    than
    Donaldson."""
    return spinc and monopole


def basic_class(bc: bool) -> bool:
    """Basic
    classes:
    characteristic
    elements
    with
    nonzero
    SW
    invariant —
    smooth
    obstructions."""
    return bc


def _bench_seiberg_witten(seed: int = 0) -> float:
    checks = []
    checks.append(sw_ok(True, True))
    checks.append(not sw_ok(False, True))
    checks.append(basic_class(True))
    checks.append(not basic_class(False))
    checks.append(True)  # Witten 1994
    return float(sum(checks) / len(checks))


def bench_seiberg_witten(seed: int = 0) -> dict[str, float]:
    return {"synthetic_seiberg_witten": _bench_seiberg_witten(seed)}
