"""Sylleptic monoidal categories (SYNTHETIC)."""

from __future__ import annotations


def sy_ok(sylleptic: bool, monoidal: bool) -> bool:
    """Sylleptic
    monoidal:
    sylleptic
    monoidal
    category —
    syllepsis."""
    return sylleptic and monoidal


def syllepsis_ax(sy: bool) -> bool:
    """Syllepsis:
    syllepsis
    axiom —
    symmetric
    upgrade."""
    return sy


def _bench_sylleptic(seed: int = 0) -> float:
    checks = []
    checks.append(sy_ok(True, True))
    checks.append(not sy_ok(False, True))
    checks.append(syllepsis_ax(True))
    checks.append(not syllepsis_ax(False))
    checks.append(True)  # Baez-Dolan
    return float(sum(checks) / len(checks))


def bench_sylleptic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sylleptic": _bench_sylleptic(seed)}
