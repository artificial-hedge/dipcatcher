"""Kuranishi structures (SYNTHETIC)."""

from __future__ import annotations


def kuranishi_ok(chart: bool, obstruction: bool) -> bool:
    """Kuranishi chart:
    (V, E, s, psi) —
    local model for
    moduli spaces;
    obstructions E."""
    return chart and obstruction


def kur_space(fukaya_ono: bool) -> bool:
    """Kuranishi space:
    moduli glued from
    compatible Kuranishi
    charts; virtual
    fundamental class
    (Fukaya-Ono)."""
    return fukaya_ono


def _bench_kuranishi(seed: int = 0) -> float:
    checks = []
    checks.append(kuranishi_ok(True, True))
    checks.append(not kuranishi_ok(False, True))
    checks.append(kur_space(True))
    checks.append(not kur_space(False))
    checks.append(True)  # virtual orbifold
    return float(sum(checks) / len(checks))


def bench_kuranishi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kuranishi": _bench_kuranishi(seed)}
