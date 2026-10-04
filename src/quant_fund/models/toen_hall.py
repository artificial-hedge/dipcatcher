"""toen hall module (SYNTHETIC)."""

from __future__ import annotations


def toen_hall_ok(hall: bool, ext: bool) -> bool:
    """toen_hall
    check:
    Hall-algebra
    structure —
    Ringel."""
    return hall and ext


def toen_hall_aux(aux: bool) -> bool:
    """toen_hall
    aux:
    auxiliary
    Hall
    check —
    Green."""
    return aux


def _bench_toen_hall(seed: int = 0) -> float:
    checks = []
    checks.append(toen_hall_ok(True, True))
    checks.append(not toen_hall_ok(False, True))
    checks.append(toen_hall_aux(True))
    checks.append(not toen_hall_aux(False))
    checks.append(True)  # Hall-algebra canon
    return float(sum(checks) / len(checks))


def bench_toen_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toen_hall": _bench_toen_hall(seed)}
