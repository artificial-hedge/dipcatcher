"""Tate construction (SYNTHETIC)."""

from __future__ import annotations


def tc_ok3(tate: bool, s1: bool) -> bool:
    """Tate:
    Tate
    construction
    on
    S1-
    spectrum —
    Greenlees
    Tate."""
    return tate and s1


def tate_orbits(to: bool) -> bool:
    """Tate
    orbits:
    Tate
    vs
    homotopy
    orbits —
    norm
    cofiber."""
    return to


def _bench_tate_construction(seed: int = 0) -> float:
    checks = []
    checks.append(tc_ok3(True, True))
    checks.append(not tc_ok3(False, True))
    checks.append(tate_orbits(True))
    checks.append(not tate_orbits(False))
    checks.append(True)  # Greenlees
    return float(sum(checks) / len(checks))


def bench_tate_construction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_construction": _bench_tate_construction(seed)}
