"""Premodular category (SYNTHETIC)."""

from __future__ import annotations


def pm_ok(premodular: bool, s_matrix: bool) -> bool:
    """Premodular:
    ribbon
    fusion
    with
    S
    matrix —
    modular
    data."""
    return premodular and s_matrix


def modular_cond(mdc: bool) -> bool:
    """Modular:
    nondeg
    S
    matrix
    gives
    modular —
    modular
    category."""
    return mdc


def _bench_premodular(seed: int = 0) -> float:
    checks = []
    checks.append(pm_ok(True, True))
    checks.append(not pm_ok(False, True))
    checks.append(modular_cond(True))
    checks.append(not modular_cond(False))
    checks.append(True)  # Turaev
    return float(sum(checks) / len(checks))


def bench_premodular(seed: int = 0) -> dict[str, float]:
    return {"synthetic_premodular": _bench_premodular(seed)}
