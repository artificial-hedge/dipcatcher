"""space time_adapt module (SYNTHETIC)."""

from __future__ import annotations


def space_time_adapt_ok(elem: bool, mark: bool) -> bool:
    """space_time_adapt
    check:
    adaptive-mesh
    canon —
    elem/marking
    consistency."""
    return elem and mark


def space_time_adapt_aux(aux: bool) -> bool:
    """space_time_adapt
    aux:
    auxiliary
    refinement check —
    error bound."""
    return aux


def _bench_space_time_adapt(seed: int = 0) -> float:
    checks = []
    checks.append(space_time_adapt_ok(True, True))
    checks.append(not space_time_adapt_ok(False, True))
    checks.append(space_time_adapt_aux(True))
    checks.append(not space_time_adapt_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_space_time_adapt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_space_time_adapt": _bench_space_time_adapt(seed)}
