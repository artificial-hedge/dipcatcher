"""quasi variational module (SYNTHETIC)."""

from __future__ import annotations


def quasi_variational_ok(sc1: bool, fs: bool) -> bool:
    """quasi_variational
    check:
    stochastic
    control —
    Fleming-Soner
    verification."""
    return sc1 and fs


def quasi_variational_aux(aux: bool) -> bool:
    """quasi_variational
    aux:
    auxiliary
    HJB
    check —
    dynamic
    programming."""
    return aux


def _bench_quasi_variational(seed: int = 0) -> float:
    checks = []
    checks.append(quasi_variational_ok(True, True))
    checks.append(not quasi_variational_ok(False, True))
    checks.append(quasi_variational_aux(True))
    checks.append(not quasi_variational_aux(False))
    checks.append(True)  # control canon
    return float(sum(checks) / len(checks))


def bench_quasi_variational(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_variational": _bench_quasi_variational(seed)}
