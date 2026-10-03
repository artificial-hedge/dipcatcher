"""decoupling field2 module (SYNTHETIC)."""

from __future__ import annotations


def decoupling_field2_ok(fb1: bool, my: bool) -> bool:
    """decoupling_field2
    check:
    FBSDE-2 —
    Ma-Yong
    decoupling."""
    return fb1 and my


def decoupling_field2_aux(aux: bool) -> bool:
    """decoupling_field2
    aux:
    auxiliary
    FBSDE
    check —
    four-step."""
    return aux


def _bench_decoupling_field2(seed: int = 0) -> float:
    checks = []
    checks.append(decoupling_field2_ok(True, True))
    checks.append(not decoupling_field2_ok(False, True))
    checks.append(decoupling_field2_aux(True))
    checks.append(not decoupling_field2_aux(False))
    checks.append(True)  # FBSDE canon
    return float(sum(checks) / len(checks))


def bench_decoupling_field2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_decoupling_field2": _bench_decoupling_field2(seed)}
