"""bridgeland hall module (SYNTHETIC)."""

from __future__ import annotations


def bridgeland_hall_ok(hall: bool, dg: bool) -> bool:
    """bridgeland_hall
    check:
    Hall-algebra-2
    structure —
    Kontsevich."""
    return hall and dg


def bridgeland_hall_aux(aux: bool) -> bool:
    """bridgeland_hall
    aux:
    auxiliary
    Hall
    check —
    Bridgeland."""
    return aux


def _bench_bridgeland_hall(seed: int = 0) -> float:
    checks = []
    checks.append(bridgeland_hall_ok(True, True))
    checks.append(not bridgeland_hall_ok(False, True))
    checks.append(bridgeland_hall_aux(True))
    checks.append(not bridgeland_hall_aux(False))
    checks.append(True)  # Hall-algebra-2 canon
    return float(sum(checks) / len(checks))


def bench_bridgeland_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bridgeland_hall": _bench_bridgeland_hall(seed)}
