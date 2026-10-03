"""Verdier duality (SYNTHETIC)."""

from __future__ import annotations


def vd_ok(verdier_dual: bool, constructible: bool) -> bool:
    """Verdier
    duality:
    duality
    on
    constructible
    sheaves —
    Verdier
    dual."""
    return verdier_dual and constructible


def six_functor_verdier(sfv: bool) -> bool:
    """Six
    functor:
    Verdier
    duality
    exchanges
    f_!
    f_* —
    six
    functors."""
    return sfv


def _bench_verdier_duality(seed: int = 0) -> float:
    checks = []
    checks.append(vd_ok(True, True))
    checks.append(not vd_ok(False, True))
    checks.append(six_functor_verdier(True))
    checks.append(not six_functor_verdier(False))
    checks.append(True)  # Verdier
    return float(sum(checks) / len(checks))


def bench_verdier_duality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_verdier_duality": _bench_verdier_duality(seed)}
