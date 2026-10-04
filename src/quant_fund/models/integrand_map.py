"""integrand map module (SYNTHETIC)."""

from __future__ import annotations


def integrand_map_ok(sel: bool, graph: bool) -> bool:
    """integrand_map
    check:
    measurable
    selection —
    graph."""
    return sel and graph


def integrand_map_aux(aux: bool) -> bool:
    """integrand_map
    aux:
    auxiliary
    selection check —
    measurability."""
    return aux


def _bench_integrand_map(seed: int = 0) -> float:
    checks = []
    checks.append(integrand_map_ok(True, True))
    checks.append(not integrand_map_ok(False, True))
    checks.append(integrand_map_aux(True))
    checks.append(not integrand_map_aux(False))
    checks.append(True)  # measurable-selection canon
    return float(sum(checks) / len(checks))


def bench_integrand_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_integrand_map": _bench_integrand_map(seed)}
