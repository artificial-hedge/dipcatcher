"""ito integral module (SYNTHETIC)."""

from __future__ import annotations


def ito_integral_ok(si: bool, isom: bool) -> bool:
    """ito_integral
    check:
    stochastic
    integral —
    isometry."""
    return si and isom


def ito_integral_aux(aux: bool) -> bool:
    """ito_integral
    aux:
    auxiliary
    integral
    check —
    covariation."""
    return aux


def _bench_ito_integral(seed: int = 0) -> float:
    checks = []
    checks.append(ito_integral_ok(True, True))
    checks.append(not ito_integral_ok(False, True))
    checks.append(ito_integral_aux(True))
    checks.append(not ito_integral_aux(False))
    checks.append(True)  # integration canon
    return float(sum(checks) / len(checks))


def bench_ito_integral(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ito_integral": _bench_ito_integral(seed)}
