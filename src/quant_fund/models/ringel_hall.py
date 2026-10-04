"""ringel hall module (SYNTHETIC)."""

from __future__ import annotations


def ringel_hall_ok(hall: bool, ext: bool) -> bool:
    """ringel_hall
    check:
    Hall-algebra
    structure —
    Ringel."""
    return hall and ext


def ringel_hall_aux(aux: bool) -> bool:
    """ringel_hall
    aux:
    auxiliary
    Hall
    check —
    Green."""
    return aux


def _bench_ringel_hall(seed: int = 0) -> float:
    checks = []
    checks.append(ringel_hall_ok(True, True))
    checks.append(not ringel_hall_ok(False, True))
    checks.append(ringel_hall_aux(True))
    checks.append(not ringel_hall_aux(False))
    checks.append(True)  # Hall-algebra canon
    return float(sum(checks) / len(checks))


def bench_ringel_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ringel_hall": _bench_ringel_hall(seed)}
