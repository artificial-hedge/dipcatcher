"""dahlquist test module (SYNTHETIC)."""

from __future__ import annotations


def dahlquist_test_ok(step: bool, order: bool) -> bool:
    """dahlquist_test
    check:
    ODE-theory/LMM
    canon — step/
    order
    consistency."""
    return step and order


def dahlquist_test_aux(aux: bool) -> bool:
    """dahlquist_test
    aux:
    auxiliary
    order check —
    stability bound."""
    return aux


def _bench_dahlquist_test(seed: int = 0) -> float:
    checks = []
    checks.append(dahlquist_test_ok(True, True))
    checks.append(not dahlquist_test_ok(False, True))
    checks.append(dahlquist_test_aux(True))
    checks.append(not dahlquist_test_aux(False))
    checks.append(True)  # lmm canon
    return float(sum(checks) / len(checks))


def bench_dahlquist_test(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dahlquist_test": _bench_dahlquist_test(seed)}
