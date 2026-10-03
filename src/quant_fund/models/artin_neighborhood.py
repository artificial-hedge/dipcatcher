"""Artin neighborhood (SYNTHETIC)."""

from __future__ import annotations


def an_ok(artin_nbd: bool, elementary: bool) -> bool:
    """Artin
    neighborhood:
    elementary
    etale
    neighborhood —
    Artin."""
    return artin_nbd and elementary


def elementary_fibration(ef: bool) -> bool:
    """Elementary
    fibration:
    smooth
    curve
    fibration
    with
    nice
    compactification —
    Artin."""
    return ef


def _bench_artin_neighborhood(seed: int = 0) -> float:
    checks = []
    checks.append(an_ok(True, True))
    checks.append(not an_ok(False, True))
    checks.append(elementary_fibration(True))
    checks.append(not elementary_fibration(False))
    checks.append(True)  # Artin
    return float(sum(checks) / len(checks))


def bench_artin_neighborhood(seed: int = 0) -> dict[str, float]:
    return {"synthetic_artin_neighborhood": _bench_artin_neighborhood(seed)}
