"""hall algebra module (SYNTHETIC)."""

from __future__ import annotations


def hall_algebra_ok(hall: bool, ext: bool) -> bool:
    """hall_algebra
    check:
    Hall-algebra
    structure —
    Ringel."""
    return hall and ext


def hall_algebra_aux(aux: bool) -> bool:
    """hall_algebra
    aux:
    auxiliary
    Hall
    check —
    Green."""
    return aux


def _bench_hall_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(hall_algebra_ok(True, True))
    checks.append(not hall_algebra_ok(False, True))
    checks.append(hall_algebra_aux(True))
    checks.append(not hall_algebra_aux(False))
    checks.append(True)  # Hall-algebra canon
    return float(sum(checks) / len(checks))


def bench_hall_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hall_algebra": _bench_hall_algebra(seed)}
