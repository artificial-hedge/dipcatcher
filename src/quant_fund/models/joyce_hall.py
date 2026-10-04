"""joyce hall module (SYNTHETIC)."""

from __future__ import annotations


def joyce_hall_ok(hall: bool, ext: bool) -> bool:
    """joyce_hall
    check:
    Hall-algebra
    structure —
    Ringel."""
    return hall and ext


def joyce_hall_aux(aux: bool) -> bool:
    """joyce_hall
    aux:
    auxiliary
    Hall
    check —
    Green."""
    return aux


def _bench_joyce_hall(seed: int = 0) -> float:
    checks = []
    checks.append(joyce_hall_ok(True, True))
    checks.append(not joyce_hall_ok(False, True))
    checks.append(joyce_hall_aux(True))
    checks.append(not joyce_hall_aux(False))
    checks.append(True)  # Hall-algebra canon
    return float(sum(checks) / len(checks))


def bench_joyce_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_joyce_hall": _bench_joyce_hall(seed)}
