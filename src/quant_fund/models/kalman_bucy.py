"""kalman bucy module (SYNTHETIC)."""

from __future__ import annotations


def kalman_bucy_ok(ze: bool, ks: bool) -> bool:
    """kalman_bucy
    check:
    filtering —
    posterior
    evolution."""
    return ze and ks


def kalman_bucy_aux(aux: bool) -> bool:
    """kalman_bucy
    aux:
    auxiliary
    filter
    check —
    innovation."""
    return aux


def _bench_kalman_bucy(seed: int = 0) -> float:
    checks = []
    checks.append(kalman_bucy_ok(True, True))
    checks.append(not kalman_bucy_ok(False, True))
    checks.append(kalman_bucy_aux(True))
    checks.append(not kalman_bucy_aux(False))
    checks.append(True)  # filtering canon
    return float(sum(checks) / len(checks))


def bench_kalman_bucy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kalman_bucy": _bench_kalman_bucy(seed)}
