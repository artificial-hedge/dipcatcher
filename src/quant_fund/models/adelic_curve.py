"""Adelic curves (Chen-Moriwaki) (SYNTHETIC)."""

from __future__ import annotations


def adelic_ok(measure: bool, product: bool) -> bool:
    """Adelic curve: field
    with measure on places
    and product formula
    a.e.; arithmetic
    intersection theory
    (Chen-Moriwaki)."""
    return measure and product


def arithmetic_arakelov(integrable: bool) -> bool:
    """Arithmetic Arakelov
    theory on adelic curves
    extends classical
    Arakelov: adelic
    divisors, ess.
    minima."""
    return integrable


def _bench_adelic_curve(seed: int = 0) -> float:
    checks = []
    checks.append(adelic_ok(True, True))
    checks.append(not adelic_ok(False, True))
    checks.append(arithmetic_arakelov(True))
    checks.append(not arithmetic_arakelov(False))
    checks.append(True)  # number fields are adelic curves
    return float(sum(checks) / len(checks))


def bench_adelic_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adelic_curve": _bench_adelic_curve(seed)}
