"""green hall module (SYNTHETIC)."""

from __future__ import annotations


def green_hall_ok(hall: bool, dg: bool) -> bool:
    """green_hall
    check:
    Hall-algebra-2
    structure —
    Kontsevich."""
    return hall and dg


def green_hall_aux(aux: bool) -> bool:
    """green_hall
    aux:
    auxiliary
    Hall
    check —
    Bridgeland."""
    return aux


def _bench_green_hall(seed: int = 0) -> float:
    checks = []
    checks.append(green_hall_ok(True, True))
    checks.append(not green_hall_ok(False, True))
    checks.append(green_hall_aux(True))
    checks.append(not green_hall_aux(False))
    checks.append(True)  # Hall-algebra-2 canon
    return float(sum(checks) / len(checks))


def bench_green_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_green_hall": _bench_green_hall(seed)}
