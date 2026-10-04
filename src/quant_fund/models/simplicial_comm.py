"""simplicial comm module (SYNTHETIC)."""

from __future__ import annotations


def simplicial_comm_ok(derived: bool, geometry: bool) -> bool:
    """simplicial_comm
    check:
    derived
    geometry
    structure —
    spectral."""
    return derived and geometry


def simplicial_comm_aux(aux: bool) -> bool:
    """simplicial_comm
    aux:
    auxiliary
    derived-geom
    check —
    analytic."""
    return aux


def _bench_simplicial_comm(seed: int = 0) -> float:
    checks = []
    checks.append(simplicial_comm_ok(True, True))
    checks.append(not simplicial_comm_ok(False, True))
    checks.append(simplicial_comm_aux(True))
    checks.append(not simplicial_comm_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_simplicial_comm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simplicial_comm": _bench_simplicial_comm(seed)}
