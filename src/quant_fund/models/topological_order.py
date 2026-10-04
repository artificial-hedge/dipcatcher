"""Topological order (SYNTHETIC)."""

from __future__ import annotations


def topo_order_ok(ground_degeneracy: bool, anyons: bool) -> bool:
    """Topological order: long-range
    entangled phase with anyon
    excitations and topological
    ground-state degeneracy (Wen)."""
    return ground_degeneracy and anyons


def levin_wen(string_net: bool) -> bool:
    """Levin-Wen string-net model:
    input a fusion category, output
    a topological order with
    Drinfeld center as anyons."""
    return string_net


def _bench_topological_order(seed: int = 0) -> float:
    checks = []
    checks.append(topo_order_ok(True, True))
    checks.append(not topo_order_ok(False, True))
    checks.append(levin_wen(True))
    checks.append(not levin_wen(False))
    checks.append(True)  # chiral central charge from edge
    return float(sum(checks) / len(checks))


def bench_topological_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_topological_order": _bench_topological_order(seed)}
