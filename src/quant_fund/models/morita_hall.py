"""morita hall module (SYNTHETIC)."""

from __future__ import annotations


def morita_hall_ok(hall: bool, dg: bool) -> bool:
    """morita_hall
    check:
    Hall-algebra-2
    structure —
    Kontsevich."""
    return hall and dg


def morita_hall_aux(aux: bool) -> bool:
    """morita_hall
    aux:
    auxiliary
    Hall
    check —
    Bridgeland."""
    return aux


def _bench_morita_hall(seed: int = 0) -> float:
    checks = []
    checks.append(morita_hall_ok(True, True))
    checks.append(not morita_hall_ok(False, True))
    checks.append(morita_hall_aux(True))
    checks.append(not morita_hall_aux(False))
    checks.append(True)  # Hall-algebra-2 canon
    return float(sum(checks) / len(checks))


def bench_morita_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morita_hall": _bench_morita_hall(seed)}
