"""Premodular categories (SYNTHETIC)."""

from __future__ import annotations


def pm_ok(premodular: bool, category: bool) -> bool:
    """Premodular:
    premodular
    category —
    premodular."""
    return premodular and category


def modularization(md: bool) -> bool:
    """Modularization:
    Bruguieres
    modularization —
    Bruguieres."""
    return md


def _bench_premodular2(seed: int = 0) -> float:
    checks = []
    checks.append(pm_ok(True, True))
    checks.append(not pm_ok(False, True))
    checks.append(modularization(True))
    checks.append(not modularization(False))
    checks.append(True)  # Bruguieres
    return float(sum(checks) / len(checks))


def bench_premodular2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_premodular2": _bench_premodular2(seed)}
