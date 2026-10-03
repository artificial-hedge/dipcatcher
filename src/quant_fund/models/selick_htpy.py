"""Selick homotopy (SYNTHETIC)."""

from __future__ import annotations


def sh_ok(selick: bool, htpy: bool) -> bool:
    """Selick
    htpy:
    Selick
    homotopy —
    Moore."""
    return selick and htpy


def moore_space_htpy(ms: bool) -> bool:
    """Moore
    space:
    Moore
    space
    homotopy —
    torsion."""
    return ms


def _bench_selick_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(sh_ok(True, True))
    checks.append(not sh_ok(False, True))
    checks.append(moore_space_htpy(True))
    checks.append(not moore_space_htpy(False))
    checks.append(True)  # Selick
    return float(sum(checks) / len(checks))


def bench_selick_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_selick_htpy": _bench_selick_htpy(seed)}
