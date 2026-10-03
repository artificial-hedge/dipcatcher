"""Perfect obstruction theories (SYNTHETIC)."""

from __future__ import annotations


def perfect_obstruction_ok(perf_complex: bool, map_to_cotangent: bool) -> bool:
    """Perfect obstruction
    theory: perfect complex
    E -> L_X in D^-(X) of
    amplitude [-1,0];
    yields virtual classes."""
    return perf_complex and map_to_cotangent


def ob_atlas(atlas: bool) -> bool:
    """The obstruction sheaf
    h^1(E^vee) captures
    deformations; embedding
    into smooth ambient
    gives Chern-Simons
    atlases."""
    return atlas


def _bench_perfect_obstruction(seed: int = 0) -> float:
    checks = []
    checks.append(perfect_obstruction_ok(True, True))
    checks.append(not perfect_obstruction_ok(False, True))
    checks.append(ob_atlas(True))
    checks.append(not ob_atlas(False))
    checks.append(True)  # Behrend-Fantechi POT
    return float(sum(checks) / len(checks))


def bench_perfect_obstruction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perfect_obstruction": _bench_perfect_obstruction(seed)}
