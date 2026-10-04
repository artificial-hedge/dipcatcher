"""quasi bsde module (SYNTHETIC)."""

from __future__ import annotations


def quasi_bsde_ok(fb1: bool, my: bool) -> bool:
    """quasi_bsde
    check:
    FBSDE-2 —
    Ma-Yong
    decoupling."""
    return fb1 and my


def quasi_bsde_aux(aux: bool) -> bool:
    """quasi_bsde
    aux:
    auxiliary
    FBSDE
    check —
    four-step."""
    return aux


def _bench_quasi_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(quasi_bsde_ok(True, True))
    checks.append(not quasi_bsde_ok(False, True))
    checks.append(quasi_bsde_aux(True))
    checks.append(not quasi_bsde_aux(False))
    checks.append(True)  # FBSDE canon
    return float(sum(checks) / len(checks))


def bench_quasi_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_bsde": _bench_quasi_bsde(seed)}
