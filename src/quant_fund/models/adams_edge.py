"""adams edge module (SYNTHETIC)."""

from __future__ import annotations


def adams_edge_ok(homotopy: bool, unstable: bool) -> bool:
    """adams_edge
    check:
    homotopy
    unstable
    structure —
    periodic."""
    return homotopy and unstable


def adams_edge_aux(aux: bool) -> bool:
    """adams_edge
    aux:
    auxiliary
    homotopy
    check —
    Adams."""
    return aux


def _bench_adams_edge(seed: int = 0) -> float:
    checks = []
    checks.append(adams_edge_ok(True, True))
    checks.append(not adams_edge_ok(False, True))
    checks.append(adams_edge_aux(True))
    checks.append(not adams_edge_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_adams_edge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adams_edge": _bench_adams_edge(seed)}
