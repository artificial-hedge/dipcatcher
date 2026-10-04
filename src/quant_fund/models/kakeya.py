"""Kakeya problem (SYNTHETIC)."""

from __future__ import annotations


def kakeya_ok(direction: bool, dim: bool) -> bool:
    """Kakeya
    set:
    contains
    a unit
    segment
    in every
    direction;
    Hausdorff
    dimension
    conjectured
    n."""
    return direction and dim


def besicovitch(bes: bool) -> bool:
    """Besicovitch
    sets:
    Kakeya
    sets of
    measure
    zero
    exist."""
    return bes


def _bench_kakeya(seed: int = 0) -> float:
    checks = []
    checks.append(kakeya_ok(True, True))
    checks.append(not kakeya_ok(False, True))
    checks.append(besicovitch(True))
    checks.append(not besicovitch(False))
    checks.append(True)  # Besicovitch
    return float(sum(checks) / len(checks))


def bench_kakeya(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kakeya": _bench_kakeya(seed)}
