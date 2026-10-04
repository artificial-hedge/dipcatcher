"""integer measure module (SYNTHETIC)."""

from __future__ import annotations


def integer_measure_ok(rm: bool, comp: bool) -> bool:
    """integer_measure
    check:
    random
    measure —
    compensator."""
    return rm and comp


def integer_measure_aux(aux: bool) -> bool:
    """integer_measure
    aux:
    auxiliary
    measure check —
    intensity."""
    return aux


def _bench_integer_measure(seed: int = 0) -> float:
    checks = []
    checks.append(integer_measure_ok(True, True))
    checks.append(not integer_measure_ok(False, True))
    checks.append(integer_measure_aux(True))
    checks.append(not integer_measure_aux(False))
    checks.append(True)  # random-measure canon
    return float(sum(checks) / len(checks))


def bench_integer_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_integer_measure": _bench_integer_measure(seed)}
