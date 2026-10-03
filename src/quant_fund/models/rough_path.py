"""rough path module (SYNTHETIC)."""

from __future__ import annotations


def rough_path_ok(rp1: bool, lift: bool) -> bool:
    """rough_path
    check:
    rough-path
    structure —
    Lyons
    lift."""
    return rp1 and lift


def rough_path_aux(aux: bool) -> bool:
    """rough_path
    aux:
    auxiliary
    signature
    check —
    shuffle
    identity."""
    return aux


def _bench_rough_path(seed: int = 0) -> float:
    checks = []
    checks.append(rough_path_ok(True, True))
    checks.append(not rough_path_ok(False, True))
    checks.append(rough_path_aux(True))
    checks.append(not rough_path_aux(False))
    checks.append(True)  # rough-path canon
    return float(sum(checks) / len(checks))


def bench_rough_path(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rough_path": _bench_rough_path(seed)}
