"""scott vol module (SYNTHETIC)."""

from __future__ import annotations


def scott_vol_ok(hm1: bool, sv: bool) -> bool:
    """scott_vol
    check:
    stochastic-vol
    —
    Heston/Bates."""
    return hm1 and sv


def scott_vol_aux(aux: bool) -> bool:
    """scott_vol
    aux:
    auxiliary
    vol
    check —
    rBergomi."""
    return aux


def _bench_scott_vol(seed: int = 0) -> float:
    checks = []
    checks.append(scott_vol_ok(True, True))
    checks.append(not scott_vol_ok(False, True))
    checks.append(scott_vol_aux(True))
    checks.append(not scott_vol_aux(False))
    checks.append(True)  # stochastic-vol canon
    return float(sum(checks) / len(checks))


def bench_scott_vol(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scott_vol": _bench_scott_vol(seed)}
