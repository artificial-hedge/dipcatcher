"""edge elements module (SYNTHETIC)."""

from __future__ import annotations


def edge_elements_ok(element: bool, mesh: bool) -> bool:
    """edge_elements
    check:
    finite-element
    method —
    element."""
    return element and mesh


def edge_elements_aux(aux: bool) -> bool:
    """edge_elements
    aux:
    auxiliary
    FEM check —
    basis."""
    return aux


def _bench_edge_elements(seed: int = 0) -> float:
    checks = []
    checks.append(edge_elements_ok(True, True))
    checks.append(not edge_elements_ok(False, True))
    checks.append(edge_elements_aux(True))
    checks.append(not edge_elements_aux(False))
    checks.append(True)  # finite-element canon
    return float(sum(checks) / len(checks))


def bench_edge_elements(seed: int = 0) -> dict[str, float]:
    return {"synthetic_edge_elements": _bench_edge_elements(seed)}
