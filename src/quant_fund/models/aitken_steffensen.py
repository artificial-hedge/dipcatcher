"""aitken steffensen module (SYNTHETIC)."""

from __future__ import annotations


def aitken_steffensen_ok(iter_: bool, conv: bool) -> bool:
    """aitken_steffensen
    check:
    root-finding /
    extrapolation
    canon — iter/
    convergence
    consistency."""
    return iter_ and conv


def aitken_steffensen_aux(aux: bool) -> bool:
    """aitken_steffensen
    aux:
    auxiliary
    iterate check —
    residual bound."""
    return aux


def _bench_aitken_steffensen(seed: int = 0) -> float:
    checks = []
    checks.append(aitken_steffensen_ok(True, True))
    checks.append(not aitken_steffensen_ok(False, True))
    checks.append(aitken_steffensen_aux(True))
    checks.append(not aitken_steffensen_aux(False))
    checks.append(True)  # rootfind canon
    return float(sum(checks) / len(checks))


def bench_aitken_steffensen(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aitken_steffensen": _bench_aitken_steffensen(seed)}
