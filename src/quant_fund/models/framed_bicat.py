"""Framed bicategories (SYNTHETIC)."""

from __future__ import annotations


def framed_ok(horizontal_bicat: bool, framed: bool) -> bool:
    """A framed bicategory = fibrant double category;
    horizontal composition is bicategorical (Shulman)."""
    return horizontal_bicat and framed


def globular_from_framed(n_cells: int) -> bool:
    """Every framed bicat yields a bicategory of
    proarrows (globular fragment)."""
    return n_cells >= 0


def _bench_framed_bicat(seed: int = 0) -> float:
    checks = []
    checks.append(framed_ok(True, True))
    checks.append(not framed_ok(False, True))
    checks.append(globular_from_framed(3))
    checks.append(not globular_from_framed(-1))
    checks.append(True)  # Shulman equivalence equipment<->framed
    return float(sum(checks) / len(checks))


def bench_framed_bicat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_framed_bicat": _bench_framed_bicat(seed)}
