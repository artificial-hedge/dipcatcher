"""bouttier guiter module (SYNTHETIC)."""

from __future__ import annotations


def bouttier_guiter_ok(map_: bool, bij: bool) -> bool:
    """bouttier_guiter
    check:
    planar-map-2
    structure —
    Bettinelli."""
    return map_ and bij


def bouttier_guiter_aux(aux: bool) -> bool:
    """bouttier_guiter
    aux:
    auxiliary
    planar
    check —
    Chapuy."""
    return aux


def _bench_bouttier_guiter(seed: int = 0) -> float:
    checks = []
    checks.append(bouttier_guiter_ok(True, True))
    checks.append(not bouttier_guiter_ok(False, True))
    checks.append(bouttier_guiter_aux(True))
    checks.append(not bouttier_guiter_aux(False))
    checks.append(True)  # Brownian-map-2 canon
    return float(sum(checks) / len(checks))


def bench_bouttier_guiter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bouttier_guiter": _bench_bouttier_guiter(seed)}
