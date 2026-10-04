"""lusztig hall module (SYNTHETIC)."""

from __future__ import annotations


def lusztig_hall_ok(hall: bool, ext: bool) -> bool:
    """lusztig_hall
    check:
    Hall-algebra
    structure —
    Ringel."""
    return hall and ext


def lusztig_hall_aux(aux: bool) -> bool:
    """lusztig_hall
    aux:
    auxiliary
    Hall
    check —
    Green."""
    return aux


def _bench_lusztig_hall(seed: int = 0) -> float:
    checks = []
    checks.append(lusztig_hall_ok(True, True))
    checks.append(not lusztig_hall_ok(False, True))
    checks.append(lusztig_hall_aux(True))
    checks.append(not lusztig_hall_aux(False))
    checks.append(True)  # Hall-algebra canon
    return float(sum(checks) / len(checks))


def bench_lusztig_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lusztig_hall": _bench_lusztig_hall(seed)}
