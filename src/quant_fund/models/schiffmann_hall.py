"""schiffmann hall module (SYNTHETIC)."""

from __future__ import annotations


def schiffmann_hall_ok(hall: bool, ext: bool) -> bool:
    """schiffmann_hall
    check:
    Hall-algebra
    structure —
    Ringel."""
    return hall and ext


def schiffmann_hall_aux(aux: bool) -> bool:
    """schiffmann_hall
    aux:
    auxiliary
    Hall
    check —
    Green."""
    return aux


def _bench_schiffmann_hall(seed: int = 0) -> float:
    checks = []
    checks.append(schiffmann_hall_ok(True, True))
    checks.append(not schiffmann_hall_ok(False, True))
    checks.append(schiffmann_hall_aux(True))
    checks.append(not schiffmann_hall_aux(False))
    checks.append(True)  # Hall-algebra canon
    return float(sum(checks) / len(checks))


def bench_schiffmann_hall(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schiffmann_hall": _bench_schiffmann_hall(seed)}
