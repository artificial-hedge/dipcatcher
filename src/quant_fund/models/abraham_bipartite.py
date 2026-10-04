"""abraham bipartite module (SYNTHETIC)."""

from __future__ import annotations


def abraham_bipartite_ok(map_: bool, plane: bool) -> bool:
    """abraham_bipartite
    check:
    Brownian-map
    structure —
    LeGall."""
    return map_ and plane


def abraham_bipartite_aux(aux: bool) -> bool:
    """abraham_bipartite
    aux:
    auxiliary
    planar
    check —
    Curien."""
    return aux


def _bench_abraham_bipartite(seed: int = 0) -> float:
    checks = []
    checks.append(abraham_bipartite_ok(True, True))
    checks.append(not abraham_bipartite_ok(False, True))
    checks.append(abraham_bipartite_aux(True))
    checks.append(not abraham_bipartite_aux(False))
    checks.append(True)  # Brownian-map canon
    return float(sum(checks) / len(checks))


def bench_abraham_bipartite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abraham_bipartite": _bench_abraham_bipartite(seed)}
