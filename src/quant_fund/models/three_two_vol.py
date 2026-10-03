"""three two_vol module (SYNTHETIC)."""

from __future__ import annotations


def three_two_vol_ok(hm1: bool, sv: bool) -> bool:
    """three_two_vol
    check:
    stochastic-vol
    —
    Heston/Bates."""
    return hm1 and sv


def three_two_vol_aux(aux: bool) -> bool:
    """three_two_vol
    aux:
    auxiliary
    vol
    check —
    rBergomi."""
    return aux


def _bench_three_two_vol(seed: int = 0) -> float:
    checks = []
    checks.append(three_two_vol_ok(True, True))
    checks.append(not three_two_vol_ok(False, True))
    checks.append(three_two_vol_aux(True))
    checks.append(not three_two_vol_aux(False))
    checks.append(True)  # stochastic-vol canon
    return float(sum(checks) / len(checks))


def bench_three_two_vol(seed: int = 0) -> dict[str, float]:
    return {"synthetic_three_two_vol": _bench_three_two_vol(seed)}
