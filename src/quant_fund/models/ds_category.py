"""Deligne-Sato categories (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(deligne: bool, category: bool) -> bool:
    """Deligne:
    Deligne
    interpolation
    category —
    Deligne
    category."""
    return deligne and category


def interpol_category(ic: bool) -> bool:
    """Interpolation:
    Deligne
    interpolation
    Rep(S_t)
    —
    interpolation."""
    return ic


def _bench_ds_category(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(interpol_category(True))
    checks.append(not interpol_category(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_ds_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ds_category": _bench_ds_category(seed)}
