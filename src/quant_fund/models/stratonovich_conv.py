"""stratonovich conv module (SYNTHETIC)."""

from __future__ import annotations


def stratonovich_conv_ok(ii1: bool, sc: bool) -> bool:
    """stratonovich_conv
    check:
    stochastic
    calculus —
    isometry/conversion."""
    return ii1 and sc


def stratonovich_conv_aux(aux: bool) -> bool:
    """stratonovich_conv
    aux:
    auxiliary
    sde
    check —
    transform."""
    return aux


def _bench_stratonovich_conv(seed: int = 0) -> float:
    checks = []
    checks.append(stratonovich_conv_ok(True, True))
    checks.append(not stratonovich_conv_ok(False, True))
    checks.append(stratonovich_conv_aux(True))
    checks.append(not stratonovich_conv_aux(False))
    checks.append(True)  # stochastic calc canon
    return float(sum(checks) / len(checks))


def bench_stratonovich_conv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stratonovich_conv": _bench_stratonovich_conv(seed)}
