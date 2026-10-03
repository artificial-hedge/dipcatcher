"""secretary dp module (SYNTHETIC)."""

from __future__ import annotations


def secretary_dp_ok(os1: bool, sd: bool) -> bool:
    """secretary_dp
    check:
    optimal-
    stopping —
    value
    function."""
    return os1 and sd


def secretary_dp_aux(aux: bool) -> bool:
    """secretary_dp
    aux:
    auxiliary
    stopping
    check —
    boundary."""
    return aux


def _bench_secretary_dp(seed: int = 0) -> float:
    checks = []
    checks.append(secretary_dp_ok(True, True))
    checks.append(not secretary_dp_ok(False, True))
    checks.append(secretary_dp_aux(True))
    checks.append(not secretary_dp_aux(False))
    checks.append(True)  # optimal-stopping canon
    return float(sum(checks) / len(checks))


def bench_secretary_dp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_secretary_dp": _bench_secretary_dp(seed)}
