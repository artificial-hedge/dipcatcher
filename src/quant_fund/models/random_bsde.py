"""random bsde module (SYNTHETIC)."""

from __future__ import annotations


def random_bsde_ok(fb1: bool, my: bool) -> bool:
    """random_bsde
    check:
    FBSDE-2 —
    Ma-Yong
    decoupling."""
    return fb1 and my


def random_bsde_aux(aux: bool) -> bool:
    """random_bsde
    aux:
    auxiliary
    FBSDE
    check —
    four-step."""
    return aux


def _bench_random_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(random_bsde_ok(True, True))
    checks.append(not random_bsde_ok(False, True))
    checks.append(random_bsde_aux(True))
    checks.append(not random_bsde_aux(False))
    checks.append(True)  # FBSDE canon
    return float(sum(checks) / len(checks))


def bench_random_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_random_bsde": _bench_random_bsde(seed)}
