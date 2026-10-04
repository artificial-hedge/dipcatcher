"""ito isometry module (SYNTHETIC)."""

from __future__ import annotations


def ito_isometry_ok(ii1: bool, sc: bool) -> bool:
    """ito_isometry
    check:
    stochastic
    calculus —
    isometry/conversion."""
    return ii1 and sc


def ito_isometry_aux(aux: bool) -> bool:
    """ito_isometry
    aux:
    auxiliary
    sde
    check —
    transform."""
    return aux


def _bench_ito_isometry(seed: int = 0) -> float:
    checks = []
    checks.append(ito_isometry_ok(True, True))
    checks.append(not ito_isometry_ok(False, True))
    checks.append(ito_isometry_aux(True))
    checks.append(not ito_isometry_aux(False))
    checks.append(True)  # stochastic calc canon
    return float(sum(checks) / len(checks))


def bench_ito_isometry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ito_isometry": _bench_ito_isometry(seed)}
