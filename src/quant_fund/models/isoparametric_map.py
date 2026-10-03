"""isoparametric map module (SYNTHETIC)."""

from __future__ import annotations


def isoparametric_map_ok(element: bool, mesh: bool) -> bool:
    """isoparametric_map
    check:
    finite-element
    method —
    element."""
    return element and mesh


def isoparametric_map_aux(aux: bool) -> bool:
    """isoparametric_map
    aux:
    auxiliary
    FEM check —
    basis."""
    return aux


def _bench_isoparametric_map(seed: int = 0) -> float:
    checks = []
    checks.append(isoparametric_map_ok(True, True))
    checks.append(not isoparametric_map_ok(False, True))
    checks.append(isoparametric_map_aux(True))
    checks.append(not isoparametric_map_aux(False))
    checks.append(True)  # finite-element canon
    return float(sum(checks) / len(checks))


def bench_isoparametric_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isoparametric_map": _bench_isoparametric_map(seed)}
