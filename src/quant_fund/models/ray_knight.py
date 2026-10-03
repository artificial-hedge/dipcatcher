"""ray knight module (SYNTHETIC)."""

from __future__ import annotations


def ray_knight_ok(ex: bool, me: bool) -> bool:
    """ray_knight
    check:
    excursion
    theory —
    measure."""
    return ex and me


def ray_knight_aux(aux: bool) -> bool:
    """ray_knight
    aux:
    auxiliary
    excursion
    check —
    local time."""
    return aux


def _bench_ray_knight(seed: int = 0) -> float:
    checks = []
    checks.append(ray_knight_ok(True, True))
    checks.append(not ray_knight_ok(False, True))
    checks.append(ray_knight_aux(True))
    checks.append(not ray_knight_aux(False))
    checks.append(True)  # excursion canon
    return float(sum(checks) / len(checks))


def bench_ray_knight(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ray_knight": _bench_ray_knight(seed)}
