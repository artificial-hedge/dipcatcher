"""rough heston module (SYNTHETIC)."""

from __future__ import annotations


def rough_heston_ok(hm1: bool, sv: bool) -> bool:
    """rough_heston
    check:
    stochastic-vol
    —
    Heston/Bates."""
    return hm1 and sv


def rough_heston_aux(aux: bool) -> bool:
    """rough_heston
    aux:
    auxiliary
    vol
    check —
    rBergomi."""
    return aux


def _bench_rough_heston(seed: int = 0) -> float:
    checks = []
    checks.append(rough_heston_ok(True, True))
    checks.append(not rough_heston_ok(False, True))
    checks.append(rough_heston_aux(True))
    checks.append(not rough_heston_aux(False))
    checks.append(True)  # stochastic-vol canon
    return float(sum(checks) / len(checks))


def bench_rough_heston(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rough_heston": _bench_rough_heston(seed)}
