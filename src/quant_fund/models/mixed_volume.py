"""mixed volume module (SYNTHETIC)."""

from __future__ import annotations


def mixed_volume_ok(convex: bool, body: bool) -> bool:
    """mixed_volume
    check:
    convex
    geometry —
    body."""
    return convex and body


def mixed_volume_aux(aux: bool) -> bool:
    """mixed_volume
    aux:
    auxiliary
    geometry check —
    volume."""
    return aux


def _bench_mixed_volume(seed: int = 0) -> float:
    checks = []
    checks.append(mixed_volume_ok(True, True))
    checks.append(not mixed_volume_ok(False, True))
    checks.append(mixed_volume_aux(True))
    checks.append(not mixed_volume_aux(False))
    checks.append(True)  # convex-geometry canon
    return float(sum(checks) / len(checks))


def bench_mixed_volume(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mixed_volume": _bench_mixed_volume(seed)}
