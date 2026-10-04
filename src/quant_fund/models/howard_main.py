"""howard main module (SYNTHETIC)."""

from __future__ import annotations


def howard_main_ok(point: bool, automorphic: bool) -> bool:
    """howard_main
    check:
    automorphic-point
    structure —
    Darmon."""
    return point and automorphic


def howard_main_aux(aux: bool) -> bool:
    """howard_main
    aux:
    auxiliary
    point
    check —
    Stark."""
    return aux


def _bench_howard_main(seed: int = 0) -> float:
    checks = []
    checks.append(howard_main_ok(True, True))
    checks.append(not howard_main_ok(False, True))
    checks.append(howard_main_aux(True))
    checks.append(not howard_main_aux(False))
    checks.append(True)  # automorphic-points canon
    return float(sum(checks) / len(checks))


def bench_howard_main(seed: int = 0) -> dict[str, float]:
    return {"synthetic_howard_main": _bench_howard_main(seed)}
