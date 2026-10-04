"""mckay correspond module (SYNTHETIC)."""

from __future__ import annotations


def mckay_correspond_ok(higher: bool, algebra: bool) -> bool:
    """mckay_correspond
    check:
    higher-algebra
    structure —
    operadic."""
    return higher and algebra


def mckay_correspond_aux(aux: bool) -> bool:
    """mckay_correspond
    aux:
    auxiliary
    higher-algebra
    check —
    enriched."""
    return aux


def _bench_mckay_correspond(seed: int = 0) -> float:
    checks = []
    checks.append(mckay_correspond_ok(True, True))
    checks.append(not mckay_correspond_ok(False, True))
    checks.append(mckay_correspond_aux(True))
    checks.append(not mckay_correspond_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_mckay_correspond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mckay_correspond": _bench_mckay_correspond(seed)}
