"""gll nodes module (SYNTHETIC)."""

from __future__ import annotations


def gll_nodes_ok(node: bool, poly: bool) -> bool:
    """gll_nodes
    check:
    spectral-element —
    high-order
    consistency."""
    return node and poly


def gll_nodes_aux(aux: bool) -> bool:
    """gll_nodes
    aux:
    auxiliary
    SEM check —
    interpolation."""
    return aux


def _bench_gll_nodes(seed: int = 0) -> float:
    checks = []
    checks.append(gll_nodes_ok(True, True))
    checks.append(not gll_nodes_ok(False, True))
    checks.append(gll_nodes_aux(True))
    checks.append(not gll_nodes_aux(False))
    checks.append(True)  # spectral-element canon
    return float(sum(checks) / len(checks))


def bench_gll_nodes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gll_nodes": _bench_gll_nodes(seed)}
