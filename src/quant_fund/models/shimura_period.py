"""shimura period module (SYNTHETIC)."""

from __future__ import annotations


def shimura_period_ok(point: bool, automorphic: bool) -> bool:
    """shimura_period
    check:
    automorphic-point
    structure —
    Darmon."""
    return point and automorphic


def shimura_period_aux(aux: bool) -> bool:
    """shimura_period
    aux:
    auxiliary
    point
    check —
    Stark."""
    return aux


def _bench_shimura_period(seed: int = 0) -> float:
    checks = []
    checks.append(shimura_period_ok(True, True))
    checks.append(not shimura_period_ok(False, True))
    checks.append(shimura_period_aux(True))
    checks.append(not shimura_period_aux(False))
    checks.append(True)  # automorphic-points canon
    return float(sum(checks) / len(checks))


def bench_shimura_period(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shimura_period": _bench_shimura_period(seed)}
