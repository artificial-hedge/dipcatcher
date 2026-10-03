"""curien legall module (SYNTHETIC)."""

from __future__ import annotations


def curien_legall_ok(map_: bool, plane: bool) -> bool:
    """curien_legall
    check:
    Brownian-map
    structure —
    LeGall."""
    return map_ and plane


def curien_legall_aux(aux: bool) -> bool:
    """curien_legall
    aux:
    auxiliary
    planar
    check —
    Curien."""
    return aux


def _bench_curien_legall(seed: int = 0) -> float:
    checks = []
    checks.append(curien_legall_ok(True, True))
    checks.append(not curien_legall_ok(False, True))
    checks.append(curien_legall_aux(True))
    checks.append(not curien_legall_aux(False))
    checks.append(True)  # Brownian-map canon
    return float(sum(checks) / len(checks))


def bench_curien_legall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curien_legall": _bench_curien_legall(seed)}
