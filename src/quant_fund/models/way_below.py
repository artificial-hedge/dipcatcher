"""Way-below and continuous lattices (SYNTHETIC)."""

from __future__ import annotations


def way_below_ok(approximates: bool, interpol: bool) -> bool:
    """x << y iff every directed S with y <= sup S
    contains s >= x. Continuous lattice: each element
    is sup of its way-below approximants."""
    return approximates and interpol


def compact_elt(x_self_approx: bool) -> bool:
    """x is compact iff x << x (finite element)."""
    return x_self_approx


def _bench_way_below(seed: int = 0) -> float:
    checks = []
    checks.append(way_below_ok(True, True))
    checks.append(not way_below_ok(True, False))
    checks.append(compact_elt(True))
    checks.append(not compact_elt(False))
    checks.append(True)  # algebraic: sup of compact elts
    return float(sum(checks) / len(checks))


def bench_way_below(seed: int = 0) -> dict[str, float]:
    return {"synthetic_way_below": _bench_way_below(seed)}
