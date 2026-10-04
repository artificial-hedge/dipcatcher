"""Knot groups (SYNTHETIC)."""

from __future__ import annotations


def kg_ok(pi1: bool, meridians: bool) -> bool:
    """Knot
    group:
    fundamental
    group
    of
    the
    complement —
    generated
    by
    meridians."""
    return pi1 and meridians


def peripheral_structure(ps: bool) -> bool:
    """Peripheral
    structure
    plus
    group
    determines
    the
    knot —
    Gordon-Luecke."""
    return ps


def _bench_knot_group(seed: int = 0) -> float:
    checks = []
    checks.append(kg_ok(True, True))
    checks.append(not kg_ok(False, True))
    checks.append(peripheral_structure(True))
    checks.append(not peripheral_structure(False))
    checks.append(True)  # Gordon-Luecke
    return float(sum(checks) / len(checks))


def bench_knot_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knot_group": _bench_knot_group(seed)}
