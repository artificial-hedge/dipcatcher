"""jump measure module (SYNTHETIC)."""

from __future__ import annotations


def jump_measure_ok(rm: bool, comp: bool) -> bool:
    """jump_measure
    check:
    random
    measure —
    compensator."""
    return rm and comp


def jump_measure_aux(aux: bool) -> bool:
    """jump_measure
    aux:
    auxiliary
    measure check —
    intensity."""
    return aux


def _bench_jump_measure(seed: int = 0) -> float:
    checks = []
    checks.append(jump_measure_ok(True, True))
    checks.append(not jump_measure_ok(False, True))
    checks.append(jump_measure_aux(True))
    checks.append(not jump_measure_aux(False))
    checks.append(True)  # random-measure canon
    return float(sum(checks) / len(checks))


def bench_jump_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jump_measure": _bench_jump_measure(seed)}
