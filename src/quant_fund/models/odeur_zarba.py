"""Odeur-Zarba (SYNTHETIC)."""

from __future__ import annotations


def oz_ok(odeur: bool, zarba: bool) -> bool:
    """Odeur
    Zarba:
    Odeur
    Zarba
    mixed
    characteristic —
    arithmetic."""
    return odeur and zarba


def mixed_characteristic(mc: bool) -> bool:
    """Mixed
    characteristic:
    mixed
    characteristic —
    zero
    p."""
    return mc


def _bench_odeur_zarba(seed: int = 0) -> float:
    checks = []
    checks.append(oz_ok(True, True))
    checks.append(not oz_ok(False, True))
    checks.append(mixed_characteristic(True))
    checks.append(not mixed_characteristic(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_odeur_zarba(seed: int = 0) -> dict[str, float]:
    return {"synthetic_odeur_zarba": _bench_odeur_zarba(seed)}
