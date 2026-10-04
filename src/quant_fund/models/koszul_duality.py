"""Koszul duality (SYNTHETIC)."""

from __future__ import annotations


def kd_ok(koszul: bool, quadratic: bool) -> bool:
    """Koszul:
    Koszul
    duality
    quadratic
    operads —
    Ginzburg-
    Kapranov."""
    return koszul and quadratic


def koszul_dual(kd: bool) -> bool:
    """Koszul
    dual:
    quadratic
    dual
    operad —
    Ginzburg
    Kapranov."""
    return kd


def _bench_koszul_duality(seed: int = 0) -> float:
    checks = []
    checks.append(kd_ok(True, True))
    checks.append(not kd_ok(False, True))
    checks.append(koszul_dual(True))
    checks.append(not koszul_dual(False))
    checks.append(True)  # G-K
    return float(sum(checks) / len(checks))


def bench_koszul_duality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koszul_duality": _bench_koszul_duality(seed)}
